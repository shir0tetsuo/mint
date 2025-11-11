import inspect
import os
import json

import inspect
import subprocess
import uuid
import threading
import uuid
import base64
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Any, Type, Tuple, Union, Optional, Dict, List, Literal, NewType

from fastapi import FastAPI, Header, HTTPException, status, BackgroundTasks, Depends
from fastapi.security import APIKeyHeader
from uvicorn import run as uvicorn_run
from pydantic import BaseModel, Field

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
APIKey = NewType('APIKey', str)
PermissionGroup = NewType('PermissionGroup', str)

global_lock = threading.RLock()
api_privkey = Path(os.path.join(Path(__file__).parent.resolve(), 'PrivKey.json'))
PRIVKEY = None
days_until_expiry = timedelta(days=30)
#cache    : dict[APIKey, dict[str, Any]]     = {}
apiserver = FastAPI(title="shadowsword.ca JSON Server")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True, scheme_name="APIKeyAuth")

class BaseResponse(BaseModel):
    status: str
    server_message: Optional[str] = None

class StatusResponse(BaseResponse):
    version: str
    user_permission_group: Optional[str] = None

def new_uuid():
    return str(uuid.uuid4())

def get_git_version():
    '''Gets the Git version (for status endpoint).'''
    try:
        return subprocess.check_output(
            ['git', 'describe', '--tags', '--always', '--dirty'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"
    
def store_privkey(private_key:bytes=os.urandom(32)):
    global global_lock
    global api_privkey
    with global_lock:
        with open(str(api_privkey), 'w', encoding='utf-8') as kf:
            json.dump({'privkey': base64.b64encode(private_key).decode('utf-8')}, kf, indent=4)
    print('A new private decryption key has been generated.')
    return private_key

def privkey():
    global PRIVKEY
    global global_lock
    global api_privkey
    if not PRIVKEY:
        try:
            if not api_privkey.exists():
                private_key = store_privkey()
                PRIVKEY = private_key
            else:
                with global_lock:
                    with api_privkey.open('r', encoding='utf-8') as kf:
                        data = json.load(kf)
                        PRIVKEY = base64.b64decode(data['privkey'])
        except Exception:
            raise
    return PRIVKEY

def create_api_key(
        label='Guest',
        permission_group='default'
    ) -> APIKey:

    key, ts_dec, id = privkey(), datetime.now().isoformat(timespec="seconds"), new_uuid()
    plaintext = f'{label}::{ts_dec}::{id}::{permission_group}'
    
    nonce:bytes=os.urandom(12)
    cipherblob = nonce + AESGCM(key).encrypt(nonce, plaintext.encode(), None)
    b64_cipher:APIKey = base64.urlsafe_b64encode(cipherblob).decode('utf-8')

    return b64_cipher

def generate_required_private():
    b64_cipher = create_api_key(label='Administrator', permission_group='admin')
    print(f'This is your new administrator X-API-Key : "{b64_cipher}" Keep it secret.')
    return

def decrypt_api_key(b64_cipher:APIKey):
    '''
    :returns tuple: `_VALID_STRUCT:bool`, `_LABEL:str`, `_PERMISSION_GROUP:str`, `_DAYS_OLD:int`
    '''

    _VALID_STRUCT = False
    _LABEL = 'Unknown'
    _PERMISSION_GROUP = 'NO_PERMISSIONS'
    _DAYS_OLD = 0
    _ID = 'NO_ID'

    def _values(): 
        print(f'Authorization Request: {_LABEL}:{_ID} with {_PERMISSION_GROUP} @ {_DAYS_OLD} Days: {_VALID_STRUCT}')
        return (_VALID_STRUCT, _LABEL, _PERMISSION_GROUP, _DAYS_OLD, _ID)
    
    try:
        decoded_blob = base64.urlsafe_b64decode(b64_cipher)
        nonce_dec, cipher_dec = decoded_blob[:12], decoded_blob[12:]
        key = privkey()
        plaintext = AESGCM(key).decrypt(nonce_dec, cipher_dec, None).decode('utf-8')
        token_parts = plaintext.split('::')

        _LABEL, _TS, _ID, _PERMISSION_GROUP = token_parts

        ts_dec = datetime.fromisoformat(_TS)
        age = datetime.now() - ts_dec

        _DAYS_OLD = age.days
        _VALID_STRUCT = True

    except Exception as e:
        print(e)

    return _values()

def Authorization(api_key: str = Depends(api_key_header)):
    global days_until_expiry
    valid_struct, label, permission_group, days, id = decrypt_api_key(api_key)
    if not valid_struct:
        message = "Unauthorized: Malformed API Key."
        if days >= days_until_expiry.days:
            message = f'Unauthorized API Key for {label}: Expired {days} days ago.'
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=message)
    
    return {
        "api_key": api_key, 
        "label": label, 
        'id': id,
        "permission_group": permission_group
    }

##############################################################
@apiserver.get("/status", response_model=StatusResponse)
def get_status(user_context: dict = Depends(Authorization)):
    global cache
    return StatusResponse(
        status="success",
        server_message=f"Hello {user_context['label']}!",
        version=get_git_version(),
        # TODO : This will have to be patched with a len() solution.
        user_permission_group=user_context['permission_group']
    )
    
#############################################################
# 💖 Server Start Block
def start_server(
        HOST='127.0.0.1',
        PORT=8884, 
    ):
    '''Start the FastAPI server using Uvicorn.'''
    global days_until_expiry

    # Generate SSL Certs / API Key
    generate_required_private()

    port = PORT
    
    print(f'Starting server at: localhost:{port}')

    uvicorn_run(
        apiserver,
        host=HOST, 
        port=port,
        log_level="info"
    )