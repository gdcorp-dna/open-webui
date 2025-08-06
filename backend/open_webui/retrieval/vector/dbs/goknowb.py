import logging
import uuid
import threading
import requests
import time
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
from cachetools import cached, TTLCache
from pydantic import BaseModel
from abc import ABC, abstractmethod
from fastapi import Request

from open_webui.retrieval.vector.main import (
    VectorDBBase,
    VectorItem,
    SearchResult,
    GetResult,
)
from open_webui.models.files import FileModel, Files
from open_webui.storage.provider import Storage
from open_webui.env import SRC_LOG_LEVELS
from gd_auth.client import AwsIamAuthTokenClient

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])

GOKNOWB_API_URL = os.environ.get("GOKNOWB_API_URL")

# GoKnowB specific enums and models
class PrincipalType(str, Enum):
    JOMAX = "jomax"
    IAM = "iam"
    ADGROUP = "adgroup"
    ANY = "*"


class ScopeType(str, Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class KBNodeType(str, Enum):
    DOCUMENT = "document"
    COLLECTION = "collection"


class SearchType(str, Enum):
    LEXICAL_AND_SEMANTIC = "lexical_and_semantic"
    SEMANTIC = "semantic"
    AUTO = "auto"


class KBStrategy(str, Enum):
    KNOWB001 = "KNOWB001"  # (default): Semantic chunking (500 tokens)
    KNOWB002 = "KNOWB002"  # No chunking
    KNOWB003 = "KNOWB003"  # Fixed-size chunking (500 tokens)
    KNOWB004 = "KNOWB004"  # Fixed-size chunking (500 tokens)
    OPENWEBUI = "OPENWEBUI" # No chunking

class QueryDocForm(BaseModel):
    collection_name: str
    query: str
    k: Optional[int] = None
    k_reranker: Optional[int] = None
    r: Optional[float] = None
    hybrid: Optional[bool] = None


@dataclass
class AllowedPrincipal:
    id: str
    type: PrincipalType
    scope: ScopeType


@dataclass
class ACL:
    allowed_principals: List[AllowedPrincipal]


@dataclass
class GoKnowbApiResponse:
    status_code: int
    data: Dict[str, Any]


class KnowledgeBaseClient:
    """
    Knowledge Base API Client for interacting with Bedrock KB.
    """

    def __init__(self):
        """Initialize the Knowledge Base API client."""
        self.base_url = GOKNOWB_API_URL.rstrip('/')
        self._token_cache = {}
        self._token_lock = threading.Lock()
        self._last_token_refresh = 0
        self._token_refresh_interval = 2400  # 40 minutes in seconds

    def _generate_token(self) -> str:
        """Generate a new JWT token."""
        # Get environment from ENVIRONMENT env var, default to dev-private
        gd_env = os.environ.get("ENVIRONMENT", "dev-private")
        
        sso_host = {
            "dev-private": "sso.dev-godaddy.com",
            "dev": "sso.dev-godaddy.com",
            "test": "sso.test-godaddy.com",
            "prod": "sso.godaddy.com",
        }.get(gd_env, "sso.dev-godaddy.com")  # Default to dev if unknown environment
        
        log.info(f"Generating SSO JWT token for environment: {gd_env}, using SSO host: {sso_host}")
        
        try:
            sso_client = AwsIamAuthTokenClient(
                sso_host=sso_host, refresh_min=45, primary_region="us-west-2", secondary_region="us-west-2"
            )
            token = sso_client.token
            log.info(f"Successfully generated SSO JWT token for environment {gd_env} using host {sso_host}: {token[:20]}...")
            return token
        except Exception as e:
            log.error(f"Failed to generate SSO JWT token for environment {gd_env} using host {sso_host}: {e}")
            raise

    def _get_cached_token(self) -> str:
        """Get a cached token or generate a new one if needed."""
        current_time = time.time()
        
        with self._token_lock:
            # Check if we have a cached token and if it's still valid
            if (self._token_cache.get('token') and 
                current_time - self._last_token_refresh < self._token_refresh_interval):
                log.debug(f"Using cached JWT token: {self._token_cache['token'][:20]}...")
                return self._token_cache['token']
            
            # Generate new token
            log.info("Cached token expired or not available, generating new token")
            new_token = self._generate_token()
            self._token_cache['token'] = new_token
            self._last_token_refresh = current_time
            return new_token

    def _refresh_token(self) -> str:
        """Force refresh the JWT token."""
        with self._token_lock:
            log.info("Forcing JWT token refresh")
            new_token = self._generate_token()
            self._token_cache['token'] = new_token
            self._last_token_refresh = time.time()
            return new_token

    def clear_token_cache(self):
        """Clear the token cache (useful for testing)."""
        with self._token_lock:
            self._token_cache.clear()
            self._last_token_refresh = 0
            log.info("Token cache cleared")

    def get_token_info(self) -> Dict[str, Any]:
        """Get information about the current token cache state."""
        with self._token_lock:
            current_time = time.time()
            token_age = current_time - self._last_token_refresh if self._last_token_refresh > 0 else None
            return {
                'has_token': 'token' in self._token_cache,
                'token_age_seconds': token_age,
                'token_refresh_interval': self._token_refresh_interval,
                'token_expires_in': max(0, self._token_refresh_interval - token_age) if token_age is not None else None
            }

    def _get_headers(self) -> Dict[str, str]:
        """Get the headers for API requests."""
        sso_jwt_token = self._get_cached_token()
        log.debug(f"Using SSO JWT token for API requests: {sso_jwt_token[:20]}...")
        headers = {
            "Authorization": f"sso-jwt {sso_jwt_token}"
        }
        return headers

    def _make_request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make an HTTP request with automatic token refresh on auth errors."""
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                headers = self._get_headers()
                kwargs['headers'] = headers
                
                response = requests.request(method, url, **kwargs)
                
                # If we get an auth error (401/403), try refreshing the token once
                if response.status_code in [401, 403] and retry_count < max_retries:
                    log.warning(f"Authentication failed (status {response.status_code}), refreshing token and retrying")
                    self._refresh_token()
                    retry_count += 1
                    continue
                
                return response
                
            except Exception as e:
                if retry_count < max_retries:
                    log.warning(f"Request failed, retrying ({retry_count + 1}/{max_retries}): {e}")
                    retry_count += 1
                    continue
                else:
                    raise
        
        # This should never be reached, but just in case
        raise Exception("Max retries exceeded")

    def _generate_request_id(self) -> str:
        """Generate a unique request ID for tracing."""
        return str(uuid.uuid4())[:8]

    def _log_request(self, method: str, url: str, data=None, files=None, request_id=None):
        """Log complete request information."""
        if not request_id:
            request_id = self._generate_request_id()

        log.debug(f"[{request_id}] === REQUEST START ===")
        log.debug(f"[{request_id}] Method: {method}")
        log.debug(f"[{request_id}] URL: {url}")

        if data:
            if isinstance(data, dict):
                log.debug(f"[{request_id}] Request Data: {data}")
            else:
                log.debug(f"[{request_id}] Request Data: {str(data)[:500]}...")

        if files:
            log.debug(f"[{request_id}] Files to Upload: {files}")

        log.debug(f"[{request_id}] === REQUEST END ===")
        return request_id

    def _handle_response(self, response: requests.Response, request_id: str = None) -> GoKnowbApiResponse:
        """Handle API response and return status code along with response JSON."""
        try:
            response_data = response.json()
        except ValueError:
            response_data = {"error": {
                "message": response.text or "No response content",
                "type": "json_parse_error"
            }}

        req_id = request_id or "unknown"
        log.debug(f"[{req_id}] === RESPONSE START ===")
        log.debug(f"[{req_id}] Status: {response.status_code} {response.reason}")
        log.debug(f"[{req_id}] URL: {response.url}")

        if not response.ok:
            if isinstance(response_data, dict) and 'error' in response_data:
                error = response_data['error']
                log.warning(f"[{req_id}] ERROR RESPONSE:")
                log.warning(f"[{req_id}]   Type: {error.get('type', 'Unknown')}")
                log.warning(f"[{req_id}]   Message: {error.get('message', 'No message')}")
                if 'error_code' in error:
                    log.warning(f"[{req_id}]   Error Code: {error.get('error_code')}")
            else:
                log.warning(f"[{req_id}] ERROR RESPONSE: {response_data}")

        log.debug(f"[{req_id}] === RESPONSE END ===")
        return GoKnowbApiResponse(status_code=response.status_code, data=response_data)

    def health_check(self) -> bool:
        """Check if the service is healthy."""
        url = f"{self.base_url}/health_check"
        request_id = self._log_request("GET", url)

        response = self._make_request_with_retry("GET", url)
        result = self._handle_response(response, request_id)
        log.info(f"Get KBNode details response status: {result.status_code}")
        log.info(f"Get KBNode details response data: {result.data}")
        return result.data.get('healthy', False)

    def create_kbnode_with_file(
            self,
            kb_node_id: str,
            resource_type: KBNodeType,
            files: Optional[List[str]] = None,
            acl: Optional[ACL] = None,
            kb_strategy: KBStrategy = KBStrategy.KNOWB001
    ) -> GoKnowbApiResponse:
        """
        Create a new KBNode with file upload.
        Args:
            kb_node_id: Base identifier for the kbnodes
            resource_type: Type of kbnodes (document or collection)
            files: Optional list of files to upload
            acl: Optional access control list for KBNode
            kb_strategy: Strategy for the KBNode (default: KBStrategy.KNOWB001)
        """

        if not files and resource_type == KBNodeType.DOCUMENT:
            raise ValueError("Files must be provided for document type KBNode")

        # Prepare form data
        data = {
            'kbNodeId': kb_node_id,
            'resourceType': resource_type.value,
            'kbStrategy': kb_strategy.value,
        }

        log.info(f"Creating KBNode with data: {data}")

        # Set default ACL if none provided
        if acl:
            data['acl'] = {
                'allowedPrincipals': [
                    {
                        'id': p.id,
                        'type': p.type.value,
                        'scope': p.scope.value
                    }
                    for p in acl.allowed_principals
                ]
            }
        else:
            # Use default ACL as per API spec
            data['acl'] = """{
                "allowedPrincipals": [
                    {
                        "type": "jomax", 
                        "id": "jomax:JOMAX_ID", 
                        "scope": "read"
                    }
                ]
            }"""

        files_data = None
        if files:
            files_data = []
            for file_full_path in files:
                try:
                    # Open file and let requests handle MIME type detection
                    # file = Files.get_file_by_id(file_id)
                    file_path = Storage.get_file(file_full_path)
                    file_path = Path(file_path)
                    files_data.append(('files', open(file_path, 'rb')))
                except IOError as e:
                    log.error(f"Failed to open file {file_path}: {e}")
                    # Close any already opened files
                    for _, opened_file in files_data:
                        opened_file.close()
                    raise ValueError(f"Cannot open file: {file_path}")

        # Log request for debugging
        url = f"{self.base_url}/v1/kbnodes"
        request_id = self._log_request("POST", url, data=data, files=files)
        
        log.info(f"Creating KB node with ID: {kb_node_id}")
        log.info(f"Resource type: {resource_type.value}")
        log.info(f"KB strategy: {kb_strategy.value}")
        log.info(f"Files to upload: {files}")

        try:
            response = self._make_request_with_retry(
                "POST",
                url,
                data=data,
                files=files_data
            )
            result = self._handle_response(response, request_id)
            log.info(f"KB node creation response status: {result.status_code}")
            log.info(f"KB node creation response data: {result.data}")
            return result
        finally:
            # Ensure files are closed
            if files_data:
                for _, opened_file in files_data:
                    try:
                        opened_file.close()
                    except Exception as e:
                        log.warning(f"Failed to close file: {e}")

    def get_kbnode_details(self, kb_node_id: str) -> GoKnowbApiResponse:
        """Get details of a specific KBNode."""
        url = f"{self.base_url}/v1/kbnodes/{kb_node_id}"
        request_id = self._log_request("GET", url)

        response = self._make_request_with_retry("GET", url)
        return self._handle_response(response, request_id)

    def delete_kbnode(self, kb_node_id: str) -> GoKnowbApiResponse:
        """Delete a specific KBNode."""
        url = f"{self.base_url}/v1/kbnodes/{kb_node_id}"
        log.info(f"Deleting KBNode: {kb_node_id} at URL: {url}")
        request_id = self._log_request("DELETE", url)

        response = self._make_request_with_retry("DELETE", url)
        result = self._handle_response(response, request_id)
        log.info(f"Delete KBNode response status: {result.status_code}")
        log.info(f"Delete KBNode response data: {result.data}")
        log.info(f"Delete KBNode result for {kb_node_id}: status={result.status_code}")
        return result

    def search_kb(
            self,
            query: str,
            kb_node_ids: List[str],
            max_results: Optional[int] = 5,
            score_threshold: Optional[float] = 0.5,
            search_type: SearchType = SearchType.AUTO
    ) -> GoKnowbApiResponse:
        """Search the knowledge base."""
        if not query or len(query.strip()) == 0:
            raise ValueError("Query cannot be empty")

        if not kb_node_ids or len(kb_node_ids) == 0:
            raise ValueError("At least one kbNodeId must be provided")

        if max_results is not None and max_results < 1:
            raise ValueError("maxNumberOfResult must be greater than 0")

        if score_threshold is not None and (score_threshold < 0.0 or score_threshold > 1.0):
            raise ValueError("scoreThreshold must be between 0 and 1")

        # Validate search_type is a proper enum
        if not isinstance(search_type, SearchType):
            raise ValueError(f"search_type must be a SearchType enum, got {type(search_type)}")

        # Prepare request body according to API specification
        request_body = {
            'query': query.strip(),
            'filter': {
                'kbNodeIds': kb_node_ids,
                'maxNumberOfResult': max_results,
                'scoreThreshold': score_threshold
            },
            'searchType': search_type.value
        }

        url = f"{self.base_url}/v1/search"
        log.info(f"Searching KB at URL: {url}")
        log.info(f"Query: {query[:100]}...")
        log.info(f"KB Node IDs: {kb_node_ids}")
        log.info(f"Max Results: {max_results}")
        log.info(f"Score Threshold: {score_threshold}")
        log.info(f"Search Type: {search_type.value}")
        request_id = self._log_request("POST", url, data=request_body)

        response = self._make_request_with_retry("POST", url, json=request_body)
        result = self._handle_response(response, request_id)
        log.info(f"Search KB response status: {result.status_code}")
        log.info(f"Search KB response data: {result.data}")
        return result

    def sync_kb(self, kb_node_id: str) -> GoKnowbApiResponse:
        """Trigger sync for a specific knowledge base node."""
        if not kb_node_id or len(kb_node_id.strip()) == 0:
            raise ValueError("KBNode ID cannot be empty")

        request_body = {
            'kbNodeId': kb_node_id.strip()
        }

        url = f"{self.base_url}/v1/sync"
        log.info(f"Triggering sync at URL: {url}")
        log.info(f"KBNode ID: {kb_node_id}")
        request_id = self._log_request("POST", url, data=request_body)

        response = self._make_request_with_retry("POST", url, json=request_body)
        return self._handle_response(response, request_id)

    def get_kbnode_status(self, kb_node_id: str) -> GoKnowbApiResponse:
        """Get the status of a specific KBNode."""
        if not kb_node_id or len(kb_node_id.strip()) == 0:
            raise ValueError("KBNode ID cannot be empty")

        url = f"{self.base_url}/v1/kbnodes/{kb_node_id}"
        log.info(f"Getting KBNode status at URL: {url}")
        log.info(f"KBNode ID: {kb_node_id}")
        request_id = self._log_request("GET", url)

        response = self._make_request_with_retry("GET", url)
        return self._handle_response(response, request_id)

    def _get_file_id_from_file_path(self, file_full_path: str) -> str:
        """Extract file ID from file path."""
        try:
            # Extract filename from path
            filename = Path(file_full_path).name
            # File ID is the part before the first underscore
            file_id = filename.split('_')[0] if '_' in filename else filename
            return file_id
        except Exception as e:
            log.warning(f"Could not extract file ID from path {file_full_path}: {e}")
            return "unknown"


class GoKnowbClient(VectorDBBase):
    """
    GoKnowB vector database client implementing VectorDBBase interface.
    
    This class provides a unified interface for GoKnowB operations,
    following the same pattern as OpenSearchClient.
    """
    
    BASE_COLLECTION_NAME = "open_webui"
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls):
        """Create or return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(GoKnowbClient, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the GoKnowB client."""
        if not self._initialized:
            self.client = KnowledgeBaseClient()
            GoKnowbClient._initialized = True
    
    @classmethod
    def get_instance(cls) -> 'GoKnowbClient':
        """Get the singleton instance of GoKnowbClient."""
        if cls._instance is None:
            return cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (useful for testing)."""
        with cls._lock:
            cls._instance = None
            cls._initialized = False

    def _update_collection_name(self, collection_name: str) -> str:
        """Update the collection name to ensure it is valid."""
        log.debug(f"_update_collection_name input: {collection_name}")
        if collection_name is None or collection_name == "":
            collection_name = self.BASE_COLLECTION_NAME
        elif self.BASE_COLLECTION_NAME is None or self.BASE_COLLECTION_NAME == "":
            raise ValueError("Base collection name is not set. Please set BASE_COLLECTION_NAME.")
        elif not collection_name.startswith(self.BASE_COLLECTION_NAME):
            collection_name = f"{self.BASE_COLLECTION_NAME}/{collection_name}"
        log.debug(f"_update_collection_name output: {collection_name}")
        return collection_name

    def _update_collection_name_full_search(self, collection_name: str) -> str:
        """Get the full search filename by appending 'full_search/'."""
        log.debug(f"_update_collection_name_full_search input: {collection_name}")
        if not collection_name.startswith(f"{self.BASE_COLLECTION_NAME}/full_search/"):
            result = self._update_collection_name(f"full_search/{collection_name}")
        else:
            result = collection_name
        log.debug(f"_update_collection_name_full_search output: {result}")
        return result

    def _create_search_result_from_response(self, goknowb_response: GoKnowbApiResponse) -> SearchResult:
        """Convert GoKnowB API response to SearchResult object."""
        if goknowb_response.status_code != 200:
            log.warning(f"Search response status code is not 200: {goknowb_response.status_code}")
            return None

        response_data = goknowb_response.data
        results = response_data.get("results", [])
        
        log.debug(f"Processing {len(results)} search results")

        ids = []
        documents = []
        metadatas = []
        distances = []

        for result in results:
            try:
                # Extract location information
                location = result.get("location", {})
                kb_node_id = location.get("kbNodeId", "")
                
                # Generate a unique ID for this result
                id = str(uuid.uuid4())
                ids.append(id)

                # Extract content information
                content = result.get("content", {})
                text = content.get("text", "")
                documents.append(text)

                # Extract score
                score = result.get("score", 0.0)
                distances.append(score)

                # Extract file information from kb_node_id and preserve all metadata
                metadata = {}
                try:
                    if kb_node_id and "/" in kb_node_id:
                        # New path structure: /{base_collection}/{collection_id}/{file_id}
                        path_parts = kb_node_id.split("/")
                        if len(path_parts) >= 3:
                            # Extract file_id from the filename (remove the actual filename part)
                            full_filename = path_parts[-1]  # Last part is the full filename
                            # The file_id is the part before the underscore in the filename
                            if '_' in full_filename:
                                file_id = full_filename.split('_')[0]
                            else:
                                file_id = full_filename  # Fallback to full filename if no underscore
                            file = Files.get_file_by_id(file_id)
                            
                            # Start with all the metadata that was stored during insertion
                            # This includes file_content, file_content_encoding, and all other fields
                            metadata = {
                                "kb_node_id": kb_node_id,
                                "file_id": file_id,
                            }
                            
                            # Add file information if available
                            if file:
                                metadata.update({
                                    "name": file.filename,
                                    "created_by": file.user_id,
                                    "source": file.filename,
                                    "filename": file.filename,
                                    "user_id": file.user_id,
                                    "file_path": file.path,
                                    # Include all the original file metadata
                                    **file.meta,
                                })
                            else:
                                log.warning(f"File with ID {file_id} not found in database")
                        else:
                            metadata = {
                                "kb_node_id": kb_node_id,
                            }
                    else:
                        metadata = {
                            "kb_node_id": kb_node_id,
                        }
                except Exception as e:
                    log.warning(f"Could not extract file metadata for kb_node_id {kb_node_id}: {e}")
                    metadata = {
                        "kb_node_id": kb_node_id,
                        "error": "Could not extract file metadata"
                    }
                
                metadatas.append(metadata)
                
            except Exception as e:
                log.error(f"Error processing search result: {e}")
                # Add placeholder data to maintain list consistency
                ids.append(str(uuid.uuid4()))
                documents.append("")
                metadatas.append({})
                distances.append(0.0)

        log.debug(f"Created search result with {len(ids)} items")
        return SearchResult(
            ids=[ids],
            documents=[documents],
            metadatas=[metadatas],
            distances=[distances]
        )

    def _create_get_result_from_response(self, goknowb_response: GoKnowbApiResponse) -> GetResult:
        """Convert GoKnowB API response to GetResult object."""
        if goknowb_response.status_code != 200:
            log.warning(f"Get response status code is not 200: {goknowb_response.status_code}")
            return None

        response_data = goknowb_response.data
        results = response_data.get("results", [])
        
        log.debug(f"Processing {len(results)} get results")

        ids = []
        documents = []
        metadatas = []

        for result in results:
            try:
                # Extract location information
                location = result.get("location", {})
                kb_node_id = location.get("kbNodeId", "")
                
                # Generate a unique ID for this result
                id = str(uuid.uuid4())
                ids.append(id)

                # Extract content information
                content = result.get("content", {})
                text = content.get("text", "")
                documents.append(text)

                # Extract file information from kb_node_id and preserve all metadata
                metadata = {}
                try:
                    if kb_node_id and "/" in kb_node_id:
                        # New path structure: /{base_collection}/{collection_id}/{file_id}
                        path_parts = kb_node_id.split("/")
                        if len(path_parts) >= 3:
                            file_id = path_parts[-1]  # Last part is the file_id
                            file = Files.get_file_by_id(file_id)
                            
                            # Start with all the metadata that was stored during insertion
                            # This includes file_content, file_content_encoding, and all other fields
                            metadata = {
                                "kb_node_id": kb_node_id,
                                "file_id": file_id,
                            }
                            
                            # Add file information if available
                            if file:
                                metadata.update({
                                    "name": file.filename,
                                    "created_by": file.user_id,
                                    "source": file.filename,
                                    "filename": file.filename,
                                    "user_id": file.user_id,
                                    "file_path": file.path,
                                    # Include all the original file metadata
                                    **file.meta,
                                })
                                
                                # Try to get the original metadata that was stored during insertion
                                # This would include file_content, file_content_encoding, etc.
                                # Since GoKnowB doesn't store this in the search results directly,
                                # we'll need to reconstruct it from the file if needed
                                try:
                                    # For now, we'll include the basic file info
                                    # In a future enhancement, we could store additional metadata
                                    # in a separate metadata field or retrieve it from the original file
                                    pass
                                except Exception as e:
                                    log.debug(f"Could not retrieve additional metadata for file {file_id}: {e}")
                            else:
                                log.warning(f"File with ID {file_id} not found in database")
                        else:
                            metadata = {
                                "kb_node_id": kb_node_id,
                            }
                    else:
                        metadata = {
                            "kb_node_id": kb_node_id,
                        }
                except Exception as e:
                    log.warning(f"Could not extract file metadata for kb_node_id {kb_node_id}: {e}")
                    metadata = {
                        "kb_node_id": kb_node_id,
                        "error": "Could not extract file metadata"
                    }
                
                metadatas.append(metadata)
                
            except Exception as e:
                log.error(f"Error processing get result: {e}")
                # Add placeholder data to maintain list consistency
                ids.append(str(uuid.uuid4()))
                documents.append("")
                metadatas.append({})

        log.info(f"Created get result with {len(ids)} items")
        return GetResult(
            ids=[ids],
            documents=[documents],
            metadatas=[metadatas]
        )

    def has_collection(self, collection_name: str) -> bool:
        """Check if the collection exists in the vector DB."""
        try:
            collection_name = self._update_collection_name(collection_name)
            log.info(f"Checking if collection exists: {collection_name}")
            result = self.client.get_kbnode_details(collection_name)
            if result.status_code==404 and result.data.get("error"):
                log.info(f"Collection {collection_name} does not exist: {result.data.get('error')}")
                return False
            elif result.status_code==200:
                log.info(f"Collection {collection_name} exists.")
                return True
            else:
                raise Exception(f" API response: {result}")
        except Exception as e:
            log.error(f"Failed to get details for collection_name {collection_name}: {e}")
            raise

    def _wait_for_kbnode_indexing(self, kb_node_id: str, node_type: str, max_wait_time: int = 300, poll_interval: int = 5) -> bool:
        """
        Wait for KB node to be indexed.
        
        Args:
            kb_node_id: The KB node ID to check
            node_type: Type of node for logging purposes (e.g., "Search", "Full Search")
            max_wait_time: Maximum time to wait in seconds (default: 300 seconds = 5 minutes)
            poll_interval: Time between status checks in seconds (default: 5 seconds)
            
        Returns:
            bool: True if indexed successfully, False if failed or timed out
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                log.info(f"Checking {node_type} KB node status: {kb_node_id}")
                status_result = self.client.get_kbnode_details(kb_node_id)
                
                if status_result.status_code == 200:
                    kb_node = status_result.data.get("kbNode", {})
                    status = kb_node.get("status")

                    
                    if status == "INDEXED":
                        log.info(f"✓ {node_type} KB node indexed successfully: {kb_node_id}")
                        return True
                    elif status == "FAILED":
                        error_message = kb_node.get("errorMessage", "Unknown error")
                        log.error(f"✗ {node_type} KB node indexing failed: {kb_node_id} - {error_message}")
                        return False
                    else:
                        log.info(f"⏳ {node_type} KB node status: {status} - {kb_node_id}")
                else:
                    log.warning(f"Failed to get {node_type} KB node status: {status_result.status_code}")
                
                time.sleep(poll_interval)
                
            except Exception as e:
                log.warning(f"Error checking {node_type} KB node status: {e}")
                time.sleep(poll_interval)
        
        log.warning(f"Timeout waiting for {node_type} KB node to be indexed: {kb_node_id}")
        return False

    def _get_file_id_from_file_path(self, file_full_path: str) -> str:
        """Extract file ID from file path."""
        try:
            # Extract filename from path
            filename = Path(file_full_path).name
            # File ID is the part before the first underscore
            file_id = filename.split('_')[0] if '_' in filename else filename
            return file_id
        except Exception as e:
            log.warning(f"Could not extract file ID from path {file_full_path}: {e}")
            return "unknown"

    def _get_upload_source_from_file(self, file_full_path: str) -> str:
        """Get the upload source from file metadata."""
        try:
            file_id = self._get_file_id_from_file_path(file_full_path)
            file_obj = Files.get_file_by_id(file_id)
            if file_obj and file_obj.meta and file_obj.meta.get('data', {}).get('upload_source'):
                return file_obj.meta['data']['upload_source']
            else:
                return "unknown"
        except Exception as e:
            log.warning(f"Could not determine upload source: {e}")
            return "unknown"

    def _time_indexing_process(self, kb_node_path: str) -> tuple[bool, float]:
        """Time the indexing process and return success status and duration."""
        indexing_start_time = time.time()
        search_indexed = self._wait_for_kbnode_indexing(kb_node_path, "Search")
        indexing_end_time = time.time()
        indexing_duration = indexing_end_time - indexing_start_time
        
        if search_indexed:
            log.info(f"✓ Indexing completed successfully in {indexing_duration:.2f} seconds")
        else:
            log.warning(f"✗ Indexing failed or timed out after {indexing_duration:.2f} seconds")
        
        return search_indexed, indexing_duration

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection from the vector DB."""
        try:
            log.info("delete_collection called with collection_name: %s", collection_name)
            #full_search_collection_name = self._update_collection_name_full_search(collection_name)
            search_collection_name = self._update_collection_name(collection_name)
            log.info(f"Deleting search collection: {search_collection_name}")
            result = self.client.delete_kbnode(search_collection_name)
            log.info(f"Search collection delete result: {result.status_code}")

            if result.status_code != 200:
                raise Exception(f" API response: {result}")
            log.info(f"Successfully deleted collection_name: {collection_name}")

        except Exception as e:
            log.error(f"Failed to delete collection_name {collection_name}: {e}")
            raise

    def delete_file(self, collection_name: str, file_full_name: str) -> None:
        """Delete a file inside collection from the vector DB."""
        try:
            log.info("delete_file called with collection_name: %s, file_full_name: %s", collection_name, file_full_name)
            search_collection_name = self._update_collection_name(collection_name)
            #full_search_collection_name = self._update_collection_name_full_search(collection_name)
            #result = self.client.delete_kbnode(full_search_collection_name+"/"+file_full_name)
            result = self.client.delete_kbnode(search_collection_name+"/"+file_full_name)
            if result.status_code != 200:
                raise Exception(f" API response: {result}")
            log.info(f"Successfully deleted collection_name: {collection_name}")

        except Exception as e:
            log.error(f"Failed to delete collection_name {collection_name}: {e}")
            raise

    def insert(self, collection_name: str, file_full_path: str) -> None:
        """Insert a list of vector items into a collection."""
        #full_search_collection_name = self._update_collection_name_full_search(collection_name)
        search_collection_name = self._update_collection_name(collection_name)
        
        try:
            log.info(f"Creating KB nodes for collection: {collection_name}")
            log.info(f"File path: {file_full_path}")
            log.info(f"Search collection KB node ID: {search_collection_name}")
            
            # Check if we have upload source information in the file metadata
            upload_source = self._get_upload_source_from_file(file_full_path)
            log.info(f"File upload source: {upload_source}")
            
            #log.info(f"Full search collection KB node ID: {full_search_collection_name}")
            
            log.info(f"Creating search KB node: {search_collection_name}")
            result = self.client.create_kbnode_with_file(
                kb_node_id=search_collection_name,
                resource_type=KBNodeType.DOCUMENT,
                files=[file_full_path],
                kb_strategy=KBStrategy.KNOWB004,
            )
            log.info(f"Create KB node result: {result}")
            if result.status_code != 202:
                raise Exception(f" API response: {result}")
            log.info(f"✓ Successfully created search KB node: {search_collection_name}")
            log.info(f"Successfully created file KB node: {search_collection_name}")

          
            if result.status_code != 202:
                raise Exception(f" API response: {result}")


            # Sync both KB nodes after creation
            log.info(f"Syncing search KB node: {search_collection_name}")
            sync_result = self.client.sync_kb("/" + search_collection_name)
            if sync_result.status_code == 202:
                log.info(f"✓ Successfully synced search KB node: {search_collection_name}")
            else:
                log.warning(f"Sync failed for search KB node {search_collection_name}: {sync_result.status_code}")

            file_name = Path(file_full_path).name
            file_id = file_name.split('_')[0] if '_' in file_name else file_name
            kb_node_path = f"{search_collection_name}/{file_name}"
            log.info(f"Checking indexing status for KB node: {kb_node_path}")
            if (upload_source == "knowledge" and "file-" not in search_collection_name) or \
                (upload_source == "chat" and "file-" in search_collection_name):
                search_indexed, indexing_duration = self._time_indexing_process(kb_node_path)
            else:
                search_indexed = True

            
            if not search_indexed:
                raise Exception("KB node failed to index within timeout period")
            else:
                log.info("✓ KB node indexed successfully")

        except Exception as e:
            log.error(f"Failed to create file {collection_name}/{file_full_path}: {e}")
            raise



    def upsert(self, collection_name: str, file_full_path: str) -> None:
        """Insert or update vector items in a collection."""
        # For GoKnowB, upsert is the same as insert since we're working with files
        self.insert(collection_name, file_full_path)

    def search(
        self, collection_name: str, vectors: List[List[Union[float, int]]], limit: int
    ) -> Optional[SearchResult]:
        """Search for similar vectors in a collection."""
        try:

            query = "semantic search query"
            
            search_collection_name = self._update_collection_name(collection_name)
            kb_node_ids = [search_collection_name]
                
            log.debug(f"Searching collections: {kb_node_ids}")
            log.debug(f"Vector dimensions: {len(vectors[0]) if vectors else 0}")
            result = self.client.search_kb(
                query=query,
                kb_node_ids=kb_node_ids,
                max_results=limit,
                score_threshold=0.0,
                search_type=SearchType.AUTO
            )
            
            log.debug(f"Search API response status: {result.status_code}")
            
            # Handle different status codes appropriately
            if result.status_code == 200:
                log.info(f"Search completed for query: {query[:50]}...")
                return self._create_search_result_from_response(result)
            elif result.status_code == 403:
                log.error(f"Authentication failed for search. Status: {result.status_code}, Response: {result.data}")
                return None
            elif result.status_code == 404:
                log.warning(f"Collection not found for search: {kb_node_ids}")
                return None
            else:
                log.error(f"Unexpected API response for search. Status: {result.status_code}, Response: {result.data}")
                return None
            
        except Exception as e:
            log.error(f"Search failed for query '{query}': {e}")
            return None

    def search_text(
        self, 
        collection_names: List[str] = None, 
        query: str = None,
        limit: int = None,
        search_type: SearchType = None,
        score_threshold: float = None
    ) -> Optional[SearchResult]:
        """Search for text in collections using GoKnowB API."""
        try:
            # Handle different parameter combinations
            if collection_names is None:
                raise ValueError("collection_names must be provided")
            
            if query is None:
                query = "semantic search query"
            
            if limit is None:
                limit = 10
                
            if search_type is None:
                search_type = SearchType.SEMANTIC
            elif not isinstance(search_type, SearchType):
                raise ValueError(f"search_type must be a SearchType enum, got {type(search_type)}")
                
            if score_threshold is None:
                score_threshold = 0.1
            
            # Convert collection names to kb_node_ids
            kb_node_ids = []
            for collection_name in collection_names:
                kb_node_ids.append(self._update_collection_name(collection_name))
                
            log.debug(f"Searching collections: {kb_node_ids}")
            log.debug(f"Query: {query}")
            log.debug(f"Search type: {search_type}")
            log.debug(f"Score threshold: {score_threshold}")
            log.debug(f"Limit: {limit}")
                
            result = self.client.search_kb(
                query=query,
                kb_node_ids=kb_node_ids,
                max_results=limit,
                score_threshold=score_threshold,
                search_type=search_type
            )
            
            log.debug(f"Search API response status: {result.status_code}")
            
            # Handle different status codes appropriately
            if result.status_code == 200:
                log.debug(f"Search completed for query: {query[:50]}...")
                return self._create_search_result_from_response(result)
            elif result.status_code == 403:
                log.error(f"Authentication failed for search. Status: {result.status_code}, Response: {result.data}")
                return None
            elif result.status_code == 404:
                log.warning(f"Collection not found for search: {kb_node_ids}")
                return None
            else:
                log.error(f"Unexpected API response for search. Status: {result.status_code}, Response: {result.data}")
                return None
            
        except Exception as e:
            log.error(f"Search failed for query '{query}': {e}")
            return None

    def query(
        self, collection_name: str, filter: Dict, limit: Optional[int] = None
    ) -> Optional[GetResult]:
        """Query vectors from a collection using metadata filter."""
        pass


    def get(self, collection_name: str) -> Optional[GetResult]:
        """Retrieve all vectors from a collection."""
        search_collection_name = self._update_collection_name(collection_name)
        log.info(f"Getting all documents from collection {collection_name}")
        log.info(f"Full search collection path: {search_collection_name}")
        
        return self.search_text(
            collection_names=[search_collection_name], 
            query="retrieve all documents",
            score_threshold=0.0
        )

    def delete(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict] = None,
    ) -> None:
        """Delete vectors by ID or filter from a collection."""
        try:
            if ids:
                log.info(f"Delete IDs from collection: {collection_name} - IDs: {ids}")
                # For GoKnowB, IDs would map to specific file_ids
                for file_id in ids:
                    self._delete_file_from_collection(collection_name, file_id)
                    
            elif filter:
                log.info(f"Delete collection: {collection_name} with filter: {filter}")
                # Extract file_id from filter
                file_id = filter.get("file_id")
                if file_id:
                    self._delete_file_from_collection(collection_name, file_id)
                else:
                    log.warning(f"No file_id found in filter: {filter}")
            else:
                log.warning("No IDs or filter provided for delete operation")
                
        except Exception as e:
            log.error(f"Failed to delete from collection {collection_name}: {e}")
            raise

    def _delete_file_from_collection(self, collection_name: str, file_id: str) -> None:
        """Delete a specific file from both regular and full search collections."""
        try:
            # Based on the logs, KB nodes are created using file-based structure
            # Create file-based paths for both regular and full search collections
            file_based_search_path = f"open_webui/file-{file_id}"
            file_obj = Files.get_file_by_id(file_id)
            if file_obj is None:
                log.warning(f"File with ID {file_id} not found")
                return
            file_name = file_obj.filename
            collection_based_search_path = f"open_webui/{collection_name}/{file_id}_{file_name}"
            
            log.info(f"Collection name: {collection_name}")
            log.info(f"Deleting file {file_id} from collection {collection_name}")
            log.info(f"File-based search path: {file_based_search_path}")
          
            try:
                result = self.client.delete_kbnode(file_based_search_path)
                if result.status_code == 200:
                    log.info(f"Successfully deleted file-based search document: {file_based_search_path}")
                elif result.status_code == 404:
                    log.warning(f"File-based search document not found: {file_based_search_path}")
                else:
                    log.warning(f"Unexpected response deleting file-based search document: {result.status_code}")
            except Exception as e:
                log.warning(f"Error deleting file-based search document: {e}")
            
            #Delete from full search collection
            try:
                result = self.client.delete_kbnode(collection_based_search_path)
                if result.status_code == 200:
                    log.info(f"Successfully deleted file-based full search document: {collection_based_search_path}")
                elif result.status_code == 404:
                    log.warning(f"File-based full search document not found: {collection_based_search_path}")
                else:
                    log.warning(f"Unexpected response deleting file-based full search document: {result.status_code}")
            except Exception as e:
                log.warning(f"Error deleting file-based full search document: {e}")
                
        except Exception as e:
            log.error(f"Failed to delete file {file_id} from collection {collection_name}: {e}")
            raise

    def reset(self) -> None:
        """Reset the vector database by removing all collections."""
        try:
            collection_name = self._update_collection_name(None)
            result = self.client.delete_kbnode(collection_name)
            if result.status_code != 200:
                raise Exception(f"API response: {result}")
            log.debug(f"Reset Successful. Deleted collection_name: {collection_name}")
        except Exception as e:
            log.error(f"Failed to delete collection_name {collection_name}: {e}")
            raise

    def health_check(self) -> bool:
        """Check if the service is healthy."""
        try:
            result = self.client.health_check()
            log.debug(f"Health check result: {result}")
            return result
        except Exception as e:
            log.error(f"Health check failed: {e}")
            raise

    def get_token_info(self) -> Dict[str, Any]:
        """Get information about the current JWT token cache state."""
        return self.client.get_token_info()

    def clear_token_cache(self):
        """Clear the JWT token cache (useful for testing)."""
        self.client.clear_token_cache()

    def save_docs_to_vector_db(
        self,
        request: Request,
        docs,
        collection_name,
        metadata: Optional[dict] = None,
        overwrite: bool = False,
        split: bool = True,
        add: bool = False,
        user=None,
    ) -> bool:
        log.debug(f"save_docs_to_vector_db collection name: {collection_name}, metadata: {metadata}, overwrite: {overwrite}, split: {split}, add: {add}")
        
        try:
            if self.has_collection(collection_name=collection_name):
                log.info(f"collection {collection_name} already exists")

                if overwrite:
                    self.delete_collection(collection_name=collection_name)
                    log.info(f"deleting existing collection {collection_name}")
                elif add is False:
                    log.info(
                        f"collection {collection_name} already exists, overwrite is False and add is False"
                    )
                    return True

            log.info(f"adding to collection {collection_name}")
            file_id = metadata.get('file_id')
            file = Files.get_file_by_id(file_id)
            log.debug(f"file path : {file.path}")
            full_file_name = f"{metadata.get('file_id')}_{metadata.get('name')}"
            self.insert(collection_name, file.path)
            return True
        except Exception as e:
            log.exception(e)
            raise e

    def query_doc_handler(self, request: Request, form_data: QueryDocForm) -> Optional[GetResult]:
        try:
            search_type = SearchType.SEMANTIC
            r = 0.0
            if request.app.state.config.ENABLE_RAG_HYBRID_SEARCH:
                search_type = SearchType.LEXICAL_AND_SEMANTIC
                r = (
                        form_data.r
                        if form_data.r
                        else request.app.state.config.RELEVANCE_THRESHOLD
                    )

            return self.search_text(
                collection_names=[form_data.collection_name], query=form_data.query,
                limit=form_data.k if form_data.k else request.app.state.config.TOP_K,
                search_type=search_type,
                score_threshold=r
            )
        except Exception as e:
            log.exception(e)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT(e),
            )


# Convenience function for getting the singleton client instance
def create_goknowb_client() -> GoKnowbClient:
    """
    Factory function to get the singleton GoKnowB client instance.
    
    Returns:
        GoKnowbClient: The singleton instance
    """
    return GoKnowbClient.get_instance() 