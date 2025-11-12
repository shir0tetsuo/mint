import inspect
import os
import json
import subprocess
import threading
import uuid
import requests
import base64
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Any, Type, Tuple, Union, Optional, Dict, List, Literal, NewType

from fastapi import FastAPI, Request, Header, HTTPException, status, BackgroundTasks, Depends
from fastapi.security import APIKeyHeader
from uvicorn import run as uvicorn_run
from pydantic import BaseModel, Field

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
APIKey = NewType('APIKey', str)
PermissionGroup = NewType('PermissionGroup', str)

lock = threading.RLock()
CASES, cases_file = None, Path(os.path.join(Path(__file__).parent.resolve(), 'cases.json'))
PRIVKEY, api_privkey = None, Path(os.path.join(Path(__file__).parent.resolve(), 'PrivKey.json'))
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

class CaseResponse(BaseResponse):
    user_id: str
    case_id: str
    request: str
    response: str
    resolution_date: str

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
    
def commit_cases():
    global lock
    global CASES
    global cases_file
    with lock:
        with open(str(cases_file), 'w', encoding='utf-8') as cf:
            json.dump(CASES, cf, indent=4)
    return

def cases():
    global lock
    global CASES
    global cases_file
    if not CASES:
        try:
            if cases_file.exists():
                with lock:
                    with cases_file.open('r', encoding='utf-8') as cf:
                        CASES = json.load(cf)
        except Exception as e:
            message = f'Error while obtaining CASES: {str(e)}'
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=message)

    return CASES or {}
    
def store_privkey(private_key:bytes=os.urandom(32)):
    global lock
    global api_privkey
    with lock:
        with open(str(api_privkey), 'w', encoding='utf-8') as kf:
            json.dump({'privkey': base64.b64encode(private_key).decode('utf-8')}, kf, indent=4)
    print('A new private decryption key has been generated.')
    return private_key

def privkey():
    global PRIVKEY
    global lock
    global api_privkey
    if not PRIVKEY:
        try:
            if not api_privkey.exists():
                PRIVKEY = store_privkey()
            else:
                with lock:
                    with api_privkey.open('r', encoding='utf-8') as kf:
                        data = json.load(kf)
                        PRIVKEY = base64.b64decode(data['privkey'])
        except Exception as e:
            message = f'Error while obtaining PRIVATE KEY for decryption: {str(e)}'
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=message)
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
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=message)
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
    return StatusResponse(
        status="success",
        server_message=f"Hello {user_context['label']}!",
        version=get_git_version(),
        # TODO : This will have to be patched with a len() solution.
        user_permission_group=user_context['permission_group']
    )
    
@apiserver.get("/case/{case_id}", response_model=CaseResponse)
def get_case(case_id, user_context: dict = Depends(Authorization)):
    global CASES

    CASES = cases()
    _case = CASES.get(case_id)

    if _case: 
        return CaseResponse(**_case)

    message = 'Case Not Found'
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)

@apiserver.post("/new_case", response_model=CaseResponse)
async def new_case(req: Request, user_context: dict = Depends(Authorization)):
    global CASES
    data = await req.json()
    case_id = new_uuid()

    CASES = cases()
    
    CASES[case_id] = {
        'status': 'Open',
        'user_id': user_context['id'],
        'case_id': case_id,
        'request': data['request'],
        'response': '',
        'resolution_date': (datetime.now() + timedelta(days=7)).isoformat(timespec='minutes')
    }

    commit_cases()
    
    return CaseResponse(**CASES[case_id])

#############################################################
# 💖 Server Start Block
def start_server(
        HOST='127.0.0.1',
        PORT=8884, 
    ):
    '''Start the FastAPI server using Uvicorn.'''
    global days_until_expiry

    # Generate API Key
    generate_required_private()

    port = PORT
    
    print(f'Starting server at: localhost:{port}')

    uvicorn_run(
        apiserver,
        host=HOST, 
        port=port,
        log_level="info"
    )

def put_on_server(
        APIKey,
        request: Optional[str] = None,
        get_or_post: Literal['GET', 'POST'] = 'GET',
        PORT=8884
    ):

    valid_struct, label, permission_group, days_old, id = decrypt_api_key(APIKey)

    global CASES
    case_id = new_uuid()

    CASES = cases()

    CASES[case_id] = {
        'status': 'Open',
        'user_id': id,
        'case_id': case_id,
        'request': request or '',
        'response': '',
        'resolution_date': (datetime.now() + timedelta(days=7)).isoformat(timespec='minutes')
    }

    commit_cases()

    return CASES[case_id]