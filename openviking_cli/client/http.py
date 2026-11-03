# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Async HTTP Client for OpenViking.

Implements BaseClient interface using HTTP calls to OpenViking Server.
"""

import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union

import httpx

from openviking.core.peer_id import normalize_peer_selector
from openviking.telemetry import TelemetryRequest, normalize_telemetry_request
from openviking.utils.search_filters import SearchContextTypeInput
from openviking_cli.client.base import BaseClient
from openviking_cli.client.identity import ClientAuthMode, OVIdentity, _identity_arguments
from openviking_cli.exceptions import (
    AbortedError,
    AlreadyExistsError,
    ConflictError,
    DeadlineExceededError,
    EmbeddingFailedError,
    FailedPreconditionError,
    InternalError,
    InvalidArgumentError,
    InvalidURIError,
    NotFoundError,
    NotInitializedError,
    OpenVikingError,
    PermissionDeniedError,
    ProcessingError,
    ResourceExhaustedError,
    SessionExpiredError,
    UnauthenticatedError,
    UnavailableError,
    UnimplementedError,
    VLMFailedError,
)
from openviking_cli.retrieve.types import FindResult
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils import get_logger, run_async
from openviking_cli.utils.config.ovcli_config import load_ovcli_config
from openviking_cli.utils.uri import VikingURI

# Error code to exception class mapping
ERROR_CODE_TO_EXCEPTION = {
    "INVALID_ARGUMENT": InvalidArgumentError,
    "INVALID_URI": InvalidURIError,
    "NOT_FOUND": NotFoundError,
    "ALREADY_EXISTS": AlreadyExistsError,
    "CONFLICT": ConflictError,
    "FAILED_PRECONDITION": FailedPreconditionError,
    "ABORTED": AbortedError,
    "UNAUTHENTICATED": UnauthenticatedError,
    "PERMISSION_DENIED": PermissionDeniedError,
    "RESOURCE_EXHAUSTED": ResourceExhaustedError,
    "UNAVAILABLE": UnavailableError,
    "INTERNAL": InternalError,
    "DEADLINE_EXCEEDED": DeadlineExceededError,
    "UNIMPLEMENTED": UnimplementedError,
    "NOT_INITIALIZED": NotInitializedError,
    "PROCESSING_ERROR": ProcessingError,
    "EMBEDDING_FAILED": EmbeddingFailedError,
    "VLM_FAILED": VLMFailedError,
    "SESSION_EXPIRED": SessionExpiredError,
    "UNKNOWN": OpenVikingError,
}

logger = get_logger(__name__)


class _HTTPObserver:
    """Observer proxy for HTTP mode.

    Provides the same interface as the local observer but fetches data via HTTP.
    """

    def __init__(self, client: "AsyncHTTPClient") -> None:
        self._client = client
        self._cache = {}

    async def _fetch_queue_status(self) -> Dict[str, Any]:
        """Fetch queue status asynchronously."""
        return await self._client._get_queue_status()

    async def _fetch_vikingdb_status(self) -> Dict[str, Any]:
        """Fetch VikingDB status asynchronously."""
        return await self._client._get_vikingdb_status()

    async def _fetch_models_status(self) -> Dict[str, Any]:
        """Fetch models status asynchronously."""
        return await self._client._get_models_status()

    async def _fetch_system_status(self) -> Dict[str, Any]:
        """Fetch system status asynchronously."""
        return await self._client._get_system_status()

    @property
    def queue(self) -> Dict[str, Any]:
        """Get queue system status (sync wrapper)."""
        return run_async(self._fetch_queue_status())

    @property
    def vikingdb(self) -> Dict[str, Any]:
        """Get VikingDB status (sync wrapper)."""
        return run_async(self._fetch_vikingdb_status())

    @property
    def models(self) -> Dict[str, Any]:
        """Get models status (VLM, Embedding, Rerank) (sync wrapper)."""
        return run_async(self._fetch_models_status())

    @property
    def system(self) -> Dict[str, Any]:
        """Get system overall status (sync wrapper)."""
        return run_async(self._fetch_system_status())

    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        status = self.system
        return status.get("is_healthy", False)


class _IdentityRequestOptions(TypedDict, total=False):
    headers: dict[str, str]


class AsyncHTTPClient(BaseClient):
    """Async HTTP Client for OpenViking Server.

    Implements BaseClient interface using HTTP calls.
    Supports auto-loading url/api_key from ovcli.conf when not provided.

    Business methods accept keyword-only ``identity=OVIdentity(...)``. These
    headers apply only to that request, including uploads. Omitting identity
    preserves constructor/profile defaults. The API key remains client-wide.

    Examples:
        # Explicit url
        client = AsyncHTTPClient(url="http://localhost:1933", api_key="key")

        # Auto-load from ~/.openviking/ovcli.conf
        client = AsyncHTTPClient()
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        account: Optional[str] = None,
        user: Optional[str] = None,
        actor_peer_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        timeout: float = 60.0,
        extra_headers: Optional[Dict[str, str]] = None,
        profile_enabled: Optional[bool] = None,
        *,
        http_client: httpx.AsyncClient | None = None,
        auth_mode: ClientAuthMode = "api_key",
    ) -> None:
        """Initialize AsyncHTTPClient.

        Args:
            url: OpenViking Server URL. If not provided, reads from ovcli.conf.
            api_key: API key for authentication. If not provided, reads from ovcli.conf.
            user_id: User identifier. If not provided, defaults to "default".
            account: Optional account identity header for trusted deployments.
            user: Optional user identity header for trusted deployments.
            actor_peer_id: Optional peer actor scope header.
            agent_id: Legacy alias for actor_peer_id.
            timeout: HTTP request timeout in seconds. Default 60.0.
            extra_headers: Additional HTTP headers to send with requests. If not provided, reads from ovcli.conf.
            http_client: Borrow an application-owned async HTTP client without changing
                its defaults. close() detaches this SDK and leaves the pool open.
            auth_mode: Match the server mode. api_key uses credential identity;
                only trusted/dev send account and user assertion headers.
        """
        if auth_mode not in ("api_key", "trusted", "dev"):
            raise ValueError("auth_mode must be api_key, trusted or dev")
        self._auth_mode = auth_mode
        effective_user = user if user is not None else user_id
        profile_load_requested = profile_enabled is None
        should_load_cli_config = profile_enabled is not False and (
            url is None
            or api_key is None
            or account is None
            or effective_user is None
            or (actor_peer_id is None and agent_id is None)
            or timeout == 60.0
            or extra_headers is None
        )
        cli_config = None
        if should_load_cli_config:
            cli_config = load_ovcli_config()
            if cli_config is not None:
                url = url or cli_config.url
                api_key = api_key or cli_config.api_key
                account = account or cli_config.account
                effective_user = effective_user or cli_config.user
                if actor_peer_id is None and agent_id is None:
                    actor_peer_id = cli_config.actor_peer_id
                    agent_id = cli_config.agent_id
                if timeout == 60.0:  # only override default with config value
                    timeout = cli_config.timeout
                if extra_headers is None:
                    extra_headers = cli_config.extra_headers
        if profile_load_requested:
            if cli_config is None:
                cli_config = load_ovcli_config()
            if cli_config is not None:
                profile_enabled = cli_config.profile
        if not url:
            raise ValueError(
                "url is required. Pass it explicitly or configure in "
                '~/.openviking/ovcli.conf (key: "url").'
            )
        self._url = url.rstrip("/")
        self._api_key = api_key
        self._account = account
        self._user_id = effective_user
        if actor_peer_id and agent_id:
            raise ValueError("actor_peer_id cannot be used with legacy agent_id")
        self._legacy_agent_id = normalize_peer_selector(None, agent_id=agent_id)
        self._actor_peer_id = normalize_peer_selector(actor_peer_id, agent_id=agent_id)
        self._user = UserIdentifier.the_default_user()
        self._timeout = timeout
        self._extra_headers = extra_headers
        self._profile_enabled = bool(profile_enabled)
        self._upload_mode = None
        if should_load_cli_config and cli_config is not None and cli_config.upload is not None:
            self._upload_mode = cli_config.upload.mode
        self._borrowed_http = http_client
        self._http: Optional[httpx.AsyncClient] = None
        self._observer: Optional[_HTTPObserver] = None

    # ============= Lifecycle =============

    def _default_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        if self._account:
            headers["X-OpenViking-Account"] = self._account
        if self._user_id:
            headers["X-OpenViking-User"] = self._user_id
        if self._actor_peer_id:
            headers["X-OpenViking-Actor-Peer"] = self._actor_peer_id
        if self._extra_headers:
            headers.update(self._extra_headers)
        if self._auth_mode == "api_key":
            headers = {
                key: value
                for key, value in headers.items()
                if key.lower() not in {"x-openviking-account", "x-openviking-user"}
            }
        return headers

    async def initialize(self) -> None:
        """Attach a borrowed pool or initialize this SDK's owned HTTP client."""
        if self._http is not None and not self._http.is_closed:
            return
        if self._borrowed_http is not None:
            if self._borrowed_http.is_closed:
                raise RuntimeError("The application-owned HTTP client is closed")
            self._http = self._borrowed_http
        else:
            self._http = httpx.AsyncClient(
                base_url=self._url,
                headers=self._default_headers(),
                timeout=self._timeout,
                params={"profile": "1"} if self._profile_enabled else None,
            )
        self._observer = _HTTPObserver(self)

    async def close(self) -> None:
        """Detach this SDK; only an owned HTTP client is closed."""
        if self._http is not None:
            try:
                if self._borrowed_http is None:
                    await self._http.aclose()
            except RuntimeError:
                pass
            finally:
                self._http = None
                self._observer = None

    # ============= Internal Helpers =============

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Apply SDK defaults per request without changing the borrowed pool."""
        if self._http is None:
            raise RuntimeError("Initialize the OV client before making a request")
        headers = httpx.Headers(self._default_headers())
        headers.update(kwargs.pop("headers", None) or {})
        if self._profile_enabled:
            kwargs["params"] = httpx.QueryParams({"profile": "1"}).merge(kwargs.get("params") or {})
        if self._auth_mode == "api_key":
            forbidden = ("X-OpenViking-Account", "X-OpenViking-User")
            for key in forbidden:
                headers.pop(key, None)
            # httpx merges borrowed client defaults back into each request.
            # Remove assertion headers from the built request, never the pool.
            inherited = getattr(self._http, "headers", {})
            if any(key in inherited for key in forbidden):
                request = self._http.build_request(
                    method,
                    self._url + "/" + path.lstrip("/"),
                    headers=headers,
                    timeout=kwargs.pop("timeout", self._timeout),
                    **kwargs,
                )
                for key in forbidden:
                    request.headers.pop(key, None)
                return await self._http.send(request)
        return await self._http.request(
            method,
            self._url + "/" + path.lstrip("/"),
            headers=headers,
            timeout=kwargs.pop("timeout", self._timeout),
            **kwargs,
        )

    def _request_options(self, identity: OVIdentity | None) -> _IdentityRequestOptions:
        """Explicit identity overrides only this request; None preserves defaults."""
        if identity is None:
            return {}
        if not isinstance(identity, OVIdentity):
            raise TypeError("identity must be an OVIdentity")
        headers = identity.headers()
        if self._auth_mode == "api_key":
            headers.pop("X-OpenViking-Account")
            headers.pop("X-OpenViking-User")
        return {"headers": headers}

    def _handle_response_data(self, response: httpx.Response) -> Dict[str, Any]:
        """Handle HTTP response and return the decoded response envelope."""
        # Gateways can reject credentials without the OV JSON error envelope.
        if response.status_code in (401, 403):
            try:
                error = response.json().get("error", {})
            except (ValueError, AttributeError):
                error = {}
            if isinstance(error, dict) and error.get("code") in (
                "UNAUTHENTICATED",
                "PERMISSION_DENIED",
            ):
                # Preserve the service's message and resource details when available.
                self._raise_exception(error)
        if response.status_code == 401:
            raise UnauthenticatedError("OpenViking request failed: HTTP 401")
        if response.status_code == 403:
            raise PermissionDeniedError("OpenViking request failed: HTTP 403")
        try:
            data = response.json()
        except Exception:
            if not response.is_success:
                raise OpenVikingError(
                    f"HTTP {response.status_code}: {response.text or 'empty response'}",
                    code="INTERNAL",
                )
            return {}
        if data.get("status") == "error":
            self._raise_exception(data.get("error", {}))
        if not response.is_success:
            raise OpenVikingError(
                data.get("detail", f"HTTP {response.status_code}"),
                code="UNKNOWN",
            )
        return data

    def _handle_response(self, response: httpx.Response) -> Any:
        """Handle HTTP response and extract result or raise exception."""
        return self._handle_response_data(response).get("result")

    @staticmethod
    def _validate_telemetry(telemetry: TelemetryRequest) -> TelemetryRequest:
        normalize_telemetry_request(telemetry)
        return telemetry

    @staticmethod
    def _normalize_target_uri(
        target_uri: Union[str, List[str]],
    ) -> Union[str, List[str]]:
        """Normalize target_uri for the search endpoints.

        Accepts either a single string or a list of strings and applies
        ``VikingURI.normalize`` to each non-empty entry.
        """
        if isinstance(target_uri, list):
            return [VikingURI.normalize(u) if u else u for u in target_uri]
        if target_uri:
            return VikingURI.normalize(target_uri)
        return target_uri

    @staticmethod
    def _attach_telemetry(result: Any, response_data: Dict[str, Any]) -> Any:
        telemetry = response_data.get("telemetry")
        if telemetry is None:
            return result

        if result is None:
            payload: Dict[str, Any] = {}
            payload["telemetry"] = telemetry
            return payload

        if isinstance(result, dict):
            result["telemetry"] = telemetry
            return result

        return result

    def _attach_legacy_agent_scope(
        self, payload: Dict[str, Any], *, identity: OVIdentity | None = None
    ) -> Dict[str, Any]:
        if identity is None and self._legacy_agent_id:
            payload["agent_id"] = self._legacy_agent_id
        return payload

    def _attach_legacy_message_scope(
        self, payload: Dict[str, Any], *, identity: OVIdentity | None = None
    ) -> Dict[str, Any]:
        if identity is not None or not self._legacy_agent_id:
            return payload
        if payload.get("peer_id") is not None:
            raise InvalidArgumentError(
                "peer_id cannot be used when client is configured with legacy agent_id"
            )
        if payload.get("role") == "assistant":
            payload["agent_id"] = self._legacy_agent_id
        return payload

    def _batch_message_payloads(
        self, messages: list[dict], *, identity: OVIdentity | None = None
    ) -> list[dict]:
        if identity is not None or not self._legacy_agent_id:
            return messages
        scoped_messages = []
        for message in messages:
            payload = dict(message)
            scoped_messages.append(self._attach_legacy_message_scope(payload))
        return scoped_messages

    def _raise_exception(self, error: Dict[str, Any]) -> None:
        """Raise appropriate exception based on error code."""
        code = error.get("code", "UNKNOWN")
        message = error.get("message", "Unknown error")
        details = error.get("details")

        exc_class = ERROR_CODE_TO_EXCEPTION.get(code, OpenVikingError)

        # Handle different exception constructors
        if exc_class == OpenVikingError:
            raise exc_class(message, code=code, details=details)
        elif exc_class in (
            InvalidArgumentError,
            FailedPreconditionError,
            ResourceExhaustedError,
            AbortedError,
            UnimplementedError,
        ):
            raise exc_class(message, details=details)
        elif exc_class == InvalidURIError:
            uri = details.get("uri", "") if details else ""
            reason = details.get("reason", "") if details else ""
            raise exc_class(uri, reason)
        elif exc_class == NotFoundError:
            resource = details.get("resource", "") if details else ""
            resource_type = details.get("type", "resource") if details else "resource"
            raise exc_class(resource, resource_type)
        elif exc_class == PermissionDeniedError:
            resource = details.get("resource") if details else None
            raise exc_class(message, resource=resource)
        elif exc_class == AlreadyExistsError:
            resource = details.get("resource", "") if details else ""
            resource_type = details.get("type", "resource") if details else "resource"
            raise exc_class(resource, resource_type)
        else:
            raise exc_class(message)

    def _zip_directory(self, dir_path: str) -> str:
        """Create a temporary zip file from a directory."""
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise ValueError(f"Path {dir_path} is not a directory")

        root = dir_path.resolve()
        temp_dir = tempfile.gettempdir()
        zip_path = Path(temp_dir) / f"temp_upload_{uuid.uuid4().hex}.zip"

        entry_count = 0
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in dir_path.rglob("*"):
                if file_path.is_symlink():
                    continue
                if file_path.is_file():
                    # Defense-in-depth: drop files whose real path escapes the selected root
                    # if traversal behavior changes or this walk implementation is replaced.
                    if not file_path.resolve().is_relative_to(root):
                        continue
                    arcname = str(file_path.relative_to(dir_path)).replace("\\", "/")
                    zipf.write(file_path, arcname=arcname)
                    entry_count += 1

        if entry_count == 0:
            logger.warning("Created empty directory upload archive for %s", dir_path)

        return str(zip_path)

    async def _upload_temp_file(self, file_path: str, *, identity: OVIdentity | None = None) -> str:
        """Upload a file to /api/v1/resources/temp_upload and return the temp_file_id."""
        with open(file_path, "rb") as f:
            files = {"file": (Path(file_path).name, f, "application/octet-stream")}
            data = None
            if self._upload_mode:
                data = {"upload_mode": self._upload_mode}
            response = await self._request(
                "POST",
                "/api/v1/resources/temp_upload",
                files=files,
                data=data,
                **self._request_options(identity),
            )
        result = self._handle_response(response)
        return result.get("temp_file_id", "")

    # ============= Resource Management =============

    async def add_resource(
        self,
        path: str,
        to: Optional[str] = None,
        parent: Optional[str] = None,
        reason: str = "",
        instruction: str = "",
        wait: bool = False,
        timeout: Optional[float] = None,
        strict: bool = False,
        ignore_dirs: Optional[str] = None,
        include: Optional[str] = None,
        exclude: Optional[str] = None,
        directly_upload_media: bool = True,
        preserve_structure: Optional[bool] = None,
        watch_interval: float = 0,
        args: Optional[Dict[str, Any]] = None,
        telemetry: TelemetryRequest = False,
        *,
        identity: OVIdentity | None = None,
    ) -> Dict[str, Any]:
        """Add resource to OpenViking."""
        telemetry = self._validate_telemetry(telemetry)
        if to and parent:
            raise ValueError("Cannot specify both 'to' and 'parent' at the same time.")

        request_data = {
            "to": to,
            "parent": parent,
            "reason": reason,
            "instruction": instruction,
            "wait": wait,
            "timeout": timeout,
            "strict": strict,
            "ignore_dirs": ignore_dirs,
            "include": include,
            "exclude": exclude,
            "directly_upload_media": directly_upload_media,
            "watch_interval": watch_interval,
            "args": args or {},
            "telemetry": telemetry,
        }
        if preserve_structure is not None:
            request_data["preserve_structure"] = preserve_structure

        path_obj = Path(path)
        if path_obj.exists():
            if path_obj.is_dir():
                source_name = path_obj.name
                request_data["source_name"] = source_name
                zip_path = self._zip_directory(path)
                try:
                    temp_file_id = await self._upload_temp_file(
                        zip_path, **_identity_arguments(identity)
                    )
                    request_data["temp_file_id"] = temp_file_id
                finally:
                    Path(zip_path).unlink(missing_ok=True)
            elif path_obj.is_file():
                request_data["source_name"] = path_obj.name
                temp_file_id = await self._upload_temp_file(path, **_identity_arguments(identity))
                request_data["temp_file_id"] = temp_file_id
            else:
                request_data["path"] = path
        else:
            request_data["path"] = path

        response = await self._request(
            "POST", "/api/v1/resources", json=request_data, **self._request_options(identity)
        )
        response_data = self._handle_response_data(response)
        return self._attach_telemetry(response_data.get("result"), response_data)

    async def batch_add_messages(
        self,
        session_id: str,
        messages: list[dict],
        telemetry: TelemetryRequest = False,
        *,
        identity: OVIdentity | None = None,
    ) -> Dict[str, Any]:
        """Add multiple messages to a session in a single request.

        Args:
            session_id: Session ID
            messages: List of message dicts, each with "role" and optionally
                      "content", "parts", "created_at", "peer_id".
            telemetry: Whether to attach operation telemetry data to the result.

        Returns:
            Result dict with session_id, message_count, and added count.
        """
        telemetry = self._validate_telemetry(telemetry)
        payload: Dict[str, Any] = {
            "messages": self._batch_message_payloads(messages, identity=identity)
        }
        if telemetry is not False:
            payload["telemetry"] = telemetry

        response = await self._request(
            "POST",
            f"/api/v1/sessions/{session_id}/messages/batch",
            json=payload,
            **self._request_options(identity),
        )
        response_data = self._handle_response_data(response)
        return self._attach_telemetry(response_data.get("result"), response_data)

    async def add_skill(
        self,
        data: Any,
        wait: bool = False,
        timeout: Optional[float] = None,
        telemetry: TelemetryRequest = False,
        *,
        identity: OVIdentity | None = None,
    ) -> Dict[str, Any]:
        """Add skill to OpenViking."""
        telemetry = self._validate_telemetry(telemetry)
        request_data = {
            "wait": wait,
            "timeout": timeout,
        }

        if isinstance(data, str):
            path_obj = Path(data)
            if path_obj.exists():
                if path_obj.is_dir():
                    zip_path = self._zip_directory(data)
                    try:
                        temp_file_id = await self._upload_temp_file(
                            zip_path, **_identity_arguments(identity)
                        )
                        request_data["temp_file_id"] = temp_file_id
                    finally:
                        Path(zip_path).unlink(missing_ok=True)
                elif path_obj.is_file():
                    temp_file_id = await self._upload_temp_file(
                        data, **_identity_arguments(identity)
                    )
                    request_data["temp_file_id"] = temp_file_id
                else:
                    request_data["data"] = data
            else:
                request_data["data"] = data
        else:
            request_data["data"] = data

        response = await self._request(
            "POST",
            "/api/v1/skills",
            json={**request_data, "telemetry": telemetry},
            **self._request_options(identity),
        )
        response_data = self._handle_response_data(response)
        return self._attach_telemetry(response_data.get("result"), response_data)

    async def wait_processed(
        self, timeout: Optional[float] = None, *, identity: OVIdentity | None = None
    ) -> Dict[str, Any]:
        """Wait for all processing to complete."""
        http_timeout = timeout if timeout else 600.0
        response = await self._request(
            "POST",
            "/api/v1/system/wait",
            json={"timeout": timeout},
            timeout=http_timeout,
            **self._request_options(identity),
        )
        return self._handle_response(response)

    # ============= File System =============

    async def ls(
        self,
        uri: str,
        simple: bool = False,
        recursive: bool = False,
        output: str = "original",
        abs_limit: int = 256,
        show_all_hidden: bool = False,
        node_limit: int = 1000,
        *,
        identity: OVIdentity | None = None,
    ) -> List[Any]:
        """List directory contents."""
        uri = VikingURI.normalize(uri)
        response = await self._request(
            "GET",
            "/api/v1/fs/ls",
            params={
                "uri": uri,
                "simple": simple,
                "recursive": recursive,
                "output": output,
                "abs_limit": abs_limit,
                "show_all_hidden": show_all_hidden,
                "node_limit": node_limit,
            },
            **self._request_options(identity),
        )
        return self._handle_response(response)

    async def tree(
        self,
        uri: str,
        output: str = "original",
        abs_limit: int = 128,
        show_all_hidden: bool = False,
        node_limit: int = 1000,
        *,
        identity: OVIdentity | None = None,
    ) -> List[Dict[str, Any]]:
        """Get directory tree."""
        uri = VikingURI.normalize(uri)
        response = await self._request(
            "GET",
            "/api/v1/fs/tree",
            params={
                "uri": uri,
                "output": output,
                "abs_limit": abs_limit,
                "show_all_hidden": show_all_hidden,
                "node_limit": node_limit,
            },
            **self._request_options(identity),
        )
        return self._handle_response(response)

    async def stat(self, uri: str, *, identity: OVIdentity | None = None) -> Dict[str, Any]:
        """Get resource status."""
        uri = VikingURI.normalize(uri)
        response = await self._request(
            "GET", "/api/v1/fs/stat", params={"uri": uri}, **self._request_options(identity)
        )
        return self._handle_response(response)

    async def mkdir(
        self, uri: str, description: Optional[str] = None, *, identity: OVIdentity | None = None
    ) -> None:
        """Create directory."""
        uri = VikingURI.normalize(uri)
        payload = {"uri": uri}
        if description is not None:
            payload["description"] = description
        response = await self._request(
            "POST", "/api/v1/fs/mkdir", json=payload, **self._request_options(identity)
        )
        self._handle_response(response)

    async def rm(
        self, uri: str, recursive: bool = False, *, identity: OVIdentity | None = None
    ) -> None:
        """Remove resource."""
        uri = VikingURI.normalize(uri)
        response = await self._request(
            "DELETE",
            "/api/v1/fs",
            params={"uri": uri, "recursive": recursive},
            **self._request_options(identity),
        )
        self._handle_response(response)

    async def mv(self, from_uri: str, to_uri: str, *, identity: OVIdentity | None = None) -> None:
        """Move resource."""
        from_uri = VikingURI.normalize(from_uri)
        to_uri = VikingURI.normalize(to_uri)
        response = await self._request(
            "POST",
            "/api/v1/fs/mv",
            json={"from_uri": from_uri, "to_uri": to_uri},
            **self._request_options(identity),
        )
        self._handle_response(response)

    # ============= Content Reading =============

    async def read(
        self, uri: str, offset: int = 0, limit: int = -1, *, identity: OVIdentity | None = None
    ) -> str:
        """Read file content.

        Args:
            uri: Viking URI
            offset: Starting line number (0-indexed). Default 0.
            limit: Number of lines to read. -1 means read to end. Default -1.
        """
        uri = VikingURI.normalize(uri)
        response = await self._request(
            "GET",
            "/api/v1/content/read",
            params={"uri": uri, "offset": offset, "limit": limit},
            **self._request_options(identity),
        )
        return self._handle_response(response)

    async def abstract(self, uri: str, *, identity: OVIdentity | None = None) -> str:
        """Read L0 abstract."""
        uri = VikingURI.normalize(uri)
        response = await self._request(
            "GET",
            "/api/v1/content/abstract",
            params={"uri": uri},
            **self._request_options(identity),
        )
        return self._handle_response(response)

    async def overview(self, uri: str, *, identity: OVIdentity | None = None) -> str:
        """Read L1 overview."""
        uri = VikingURI.normalize(uri)
        response = await self._request(
            "GET",
            "/api/v1/content/overview",
            params={"uri": uri},
            **self._request_options(identity),
        )
        return self._handle_response(response)

    async def write(
        self,
        uri: str,
        content: str,
        mode: str = "replace",
        wait: bool = False,
        timeout: Optional[float] = None,
        telemetry: TelemetryRequest = False,
        *,
        identity: OVIdentity | None = None,
    ) -> Dict[str, Any]:
        """Write text content to an existing file and refresh semantics/vectors."""
        telemetry = self._validate_telemetry(telemetry)
        uri = VikingURI.normalize(uri)
        response = await self._request(
            "POST",
            "/api/v1/content/write",
            json={
                "uri": uri,
                "content": content,
                "mode": mode,
                "wait": wait,
                "timeout": timeout,
                "telemetry": telemetry,
            },
            **self._request_options(identity),
        )
        response_data = self._handle_response_data(response)
        return self._attach_telemetry(response_data.get("result") or {}, response_data)

    # ============= Search =============

    async def find(
        self,
        query: str,
        target_uri: Union[str, List[str]] = "",
        limit: int = 10,
        node_limit: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None,
        context_type: Optional[SearchContextTypeInput] = None,
        telemetry: TelemetryRequest = False,
        *,
        identity: OVIdentity | None = None,
    ) -> FindResult:
        """Semantic search without session context."""
        telemetry = self._validate_telemetry(telemetry)
        normalized_target = self._normalize_target_uri(target_uri)
        actual_limit = node_limit if node_limit is not None else limit
        payload = self._attach_legacy_agent_scope(
            {
                "query": query,
                "target_uri": normalized_target,
                "limit": actual_limit,
                "score_threshold": score_threshold,
                "filter": filter,
                "context_type": context_type,
                "telemetry": telemetry,
            },
            identity=identity,
        )
        response = await self._request(
            "POST", "/api/v1/search/find", json=payload, **self._request_options(identity)
        )
        response_data = self._handle_response_data(response)
        return FindResult.from_dict(response_data.get("result") or {})

    async def search(
        self,
        query: str,
        target_uri: Union[str, List[str]] = "",
        session: Optional[Any] = None,
        session_id: Optional[str] = None,
        limit: int = 10,
        node_limit: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None,
        context_type: Optional[SearchContextTypeInput] = None,
        telemetry: TelemetryRequest = False,
        *,
        identity: OVIdentity | None = None,
    ) -> FindResult:
        """Semantic search with optional session context."""
        telemetry = self._validate_telemetry(telemetry)
        normalized_target = self._normalize_target_uri(target_uri)
        actual_limit = node_limit if node_limit is not None else limit
        sid = session_id or (session.session_id if session else None)
        payload = self._attach_legacy_agent_scope(
            {
                "query": query,
                "target_uri": normalized_target,
                "session_id": sid,
                "limit": actual_limit,
                "score_threshold": score_threshold,
                "filter": filter,
                "context_type": context_type,
                "telemetry": telemetry,
            },
            identity=identity,
        )
        response = await self._request(
            "POST", "/api/v1/search/search", json=payload, **self._request_options(identity)
        )
        response_data = self._handle_response_data(response)
        return FindResult.from_dict(response_data.get("result") or {})

    async def grep(
        self,
        uri: str,
        pattern: str,
        case_insensitive: bool = False,
        node_limit: Optional[int] = None,
        exclude_uri: Optional[str] = None,
        *,
        identity: OVIdentity | None = None,
    ) -> Dict[str, Any]:
        """Content search with pattern."""
        uri = VikingURI.normalize(uri)
        request_json = {
            "uri": uri,
            "pattern": pattern,
            "case_insensitive": case_insensitive,
        }
        if node_limit is not None:
            request_json["node_limit"] = node_limit
        if exclude_uri is not None:
            request_json["exclude_uri"] = VikingURI.normalize(exclude_uri)
        response = await self._request(
            "POST", "/api/v1/search/grep", json=request_json, **self._request_options(identity)
        )
        return self._handle_response(response)

    async def glob(
        self, pattern: str, uri: str = "viking://", *, identity: OVIdentity | None = None
    ) -> Dict[str, Any]:
        """File pattern matching."""
        uri = VikingURI.normalize(uri)
        response = await self._request(
            "POST",
            "/api/v1/search/glob",
            json={"pattern": pattern, "uri": uri},
            **self._request_options(identity),
        )
        return self._handle_response(response)

    # ============= Relations =============

    async def relations(self, uri: str, *, identity: OVIdentity | None = None) -> List[Any]:
        """Get relations for a resource."""
        uri = VikingURI.normalize(uri)
        response = await self._request(
            "GET", "/api/v1/relations", params={"uri": uri}, **self._request_options(identity)
        )
        return self._handle_response(response)

    async def link(
        self,
        from_uri: str,
        to_uris: Union[str, List[str]],
        reason: str = "",
        *,
        identity: OVIdentity | None = None,
    ) -> None:
        """Create link between resources."""
        from_uri = VikingURI.normalize(from_uri)
        if isinstance(to_uris, str):
            to_uris = VikingURI.normalize(to_uris)
        else:
            to_uris = [VikingURI.normalize(u) for u in to_uris]
        response = await self._request(
            "POST",
            "/api/v1/relations/link",
            json={"from_uri": from_uri, "to_uris": to_uris, "reason": reason},
            **self._request_options(identity),
        )
        self._handle_response(response)

    async def unlink(
        self, from_uri: str, to_uri: str, *, identity: OVIdentity | None = None
    ) -> None:
        """Remove link between resources."""
        from_uri = VikingURI.normalize(from_uri)
        to_uri = VikingURI.normalize(to_uri)
        response = await self._request(
            "DELETE",
            "/api/v1/relations/link",
            json={"from_uri": from_uri, "to_uri": to_uri},
            **self._request_options(identity),
        )
        self._handle_response(response)

    # ============= Sessions =============

    async def create_session(
        self,
        session_id: Optional[str] = None,
        telemetry: TelemetryRequest = False,
        memory_policy: Optional[Dict[str, Any]] = None,
        *,
        identity: OVIdentity | None = None,
    ) -> Dict[str, Any]:
        """Create a new session.

        Args:
            session_id: Optional session ID. If provided, creates a session with the given ID.
                       If None, creates a new session with auto-generated ID.
        """
        telemetry = self._validate_telemetry(telemetry)
        json_body: Dict[str, Any] = {}
        if session_id is not None:
            json_body["session_id"] = session_id
        if memory_policy is not None:
            json_body["memory_policy"] = memory_policy
        if telemetry is not False:
            json_body["telemetry"] = telemetry
        response = await self._request(
            "POST", "/api/v1/sessions", json=json_body, **self._request_options(identity)
        )
        response_data = self._handle_response_data(response)
        return self._attach_telemetry(response_data.get("result"), response_data)

    async def list_sessions(self, *, identity: OVIdentity | None = None) -> List[Any]:
        """List all sessions."""
        response = await self._request("GET", "/api/v1/sessions", **self._request_options(identity))
        return self._handle_response(response)

    async def get_session(
        self, session_id: str, *, auto_create: bool = False, identity: OVIdentity | None = None
    ) -> Dict[str, Any]:
        """Get session details."""
        params = {}
        if auto_create:
            params["auto_create"] = "true"
        response = await self._request(
            "GET",
            f"/api/v1/sessions/{session_id}",
            params=params,
            **self._request_options(identity),
        )
        return self._handle_response(response)

    async def get_session_context(
        self, session_id: str, token_budget: int = 128_000, *, identity: OVIdentity | None = None
    ) -> Dict[str, Any]:
        """Get assembled session context."""
        response = await self._request(
            "GET",
            f"/api/v1/sessions/{session_id}/context",
            params={"token_budget": token_budget},
            **self._request_options(identity),
        )
        return self._handle_response(response)

    async def get_session_archive(
        self, session_id: str, archive_id: str, *, identity: OVIdentity | None = None
    ) -> Dict[str, Any]:
        """Get one completed archive for a session."""
        response = await self._request(
            "GET",
            f"/api/v1/sessions/{session_id}/archives/{archive_id}",
            **self._request_options(identity),
        )
        return self._handle_response(response)

    async def read_tool_result(
        self,
        session_id: str,
        tool_result_id: str,
        *,
        offset: int = 0,
        limit: int = 20_000,
        identity: OVIdentity | None = None,
    ) -> Dict[str, Any]:
        """Read Unicode characters from a session's externalized tool result."""
        from urllib.parse import quote

        response = await self._request(
            "GET",
            f"/api/v1/sessions/{quote(session_id, safe='')}/tool-results/{quote(tool_result_id, safe='')}",
            params={"offset": offset, "limit": limit},
            **self._request_options(identity),
        )
        return self._handle_response(response)

    async def search_tool_result(
        self,
        session_id: str,
        tool_result_id: str,
        *,
        query: str,
        limit: int = 20,
        identity: OVIdentity | None = None,
    ) -> Dict[str, Any]:
        """Search within a session's externalized tool result."""
        from urllib.parse import quote

        response = await self._request(
            "GET",
            f"/api/v1/sessions/{quote(session_id, safe='')}/tool-results/{quote(tool_result_id, safe='')}/search",
            params={"q": query, "limit": limit},
            **self._request_options(identity),
        )
        return self._handle_response(response)

    async def list_tool_results(
        self, session_id: str, *, limit: int = 50, identity: OVIdentity | None = None
    ) -> Dict[str, Any]:
        """List references in the current session."""
        from urllib.parse import quote

        response = await self._request(
            "GET",
            f"/api/v1/sessions/{quote(session_id, safe='')}/tool-results",
            params={"limit": limit},
            **self._request_options(identity),
        )
        return self._handle_response(response)

    async def delete_session(self, session_id: str, *, identity: OVIdentity | None = None) -> None:
        """Delete a session."""
        response = await self._request(
            "DELETE", f"/api/v1/sessions/{session_id}", **self._request_options(identity)
        )
        self._handle_response(response)

    async def get_task(
        self, task_id: str, *, identity: OVIdentity | None = None
    ) -> Optional[Dict[str, Any]]:
        """Query background task status.

        Args:
            task_id: Task ID (returned by commit)

        Returns:
            Task info dict, or None if not found
        """
        response = await self._request(
            "GET",
            f"/api/v1/tasks/{task_id}",
            params=(
                {}
                if identity is None
                else {"account_id": identity.account_id, "user_id": identity.user_id}
            ),
            **self._request_options(identity),
        )
        if response.status_code == 404:
            return None
        return self._handle_response(response)

    async def list_tasks(
        self,
        *,
        task_type: str | None = None,
        status: str | None = None,
        resource_id: str | None = None,
        limit: int = 50,
        identity: OVIdentity | None = None,
    ) -> list[dict[str, Any]]:
        """Find persisted tasks for a session after rebuilding a Run's SDK."""
        params: dict[str, Any] = {"limit": limit}
        params.update(
            {
                key: value
                for key, value in {
                    "task_type": task_type,
                    "status": status,
                    "resource_id": resource_id,
                }.items()
                if value is not None
            }
        )
        if identity is not None:
            params.update(account_id=identity.account_id, user_id=identity.user_id)
        response = await self._request(
            "GET", "/api/v1/tasks", params=params, **self._request_options(identity)
        )
        result = self._handle_response(response)
        if not isinstance(result, list):
            raise ValueError("Task list response must be a list")
        return result

    async def commit_session(
        self,
        session_id: str,
        telemetry: TelemetryRequest = False,
        *,
        keep_recent_count: int = 0,
        identity: OVIdentity | None = None,
    ) -> Dict[str, Any]:
        """Commit a session (archive and extract memories)."""
        telemetry = self._validate_telemetry(telemetry)
        payload: Dict[str, Any] = {
            "keep_recent_count": keep_recent_count,
            "telemetry": telemetry,
        }
        response = await self._request(
            "POST",
            f"/api/v1/sessions/{session_id}/commit",
            json=payload,
            **self._request_options(identity),
        )
        response_data = self._handle_response_data(response)
        return self._attach_telemetry(response_data.get("result"), response_data)

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str | None = None,
        parts: list[dict] | None = None,
        created_at: str | None = None,
        peer_id: str | None = None,
        telemetry: TelemetryRequest = False,
        *,
        identity: OVIdentity | None = None,
        message_kind: str | None = None,
        source_message_ids: list[str] | None = None,
    ) -> Dict[str, Any]:
        """Add a message to a session.

        Args:
            session_id: Session ID
            role: Message role ("user" or "assistant")
            content: Text content (simple mode, backward compatible)
            parts: Parts array (full Part support mode)
            created_at: Message creation time (ISO format string)
            peer_id: Optional stable interaction peer identity.

        If both content and parts are provided, parts takes precedence.
        """
        telemetry = self._validate_telemetry(telemetry)
        payload: Dict[str, Any] = {"role": role}
        if parts is not None:
            payload["parts"] = parts
        elif content is not None:
            payload["content"] = content
        else:
            raise ValueError("Either content or parts must be provided")

        if created_at is not None:
            payload["created_at"] = created_at
        if peer_id is not None:
            payload["peer_id"] = peer_id
        if message_kind is not None:
            payload["message_kind"] = message_kind
        if source_message_ids is not None:
            payload["source_message_ids"] = source_message_ids
        if telemetry is not False:
            payload["telemetry"] = telemetry
        payload = self._attach_legacy_message_scope(payload, identity=identity)

        response = await self._request(
            "POST",
            f"/api/v1/sessions/{session_id}/messages",
            json=payload,
            **self._request_options(identity),
        )
        response_data = self._handle_response_data(response)
        return self._attach_telemetry(response_data.get("result"), response_data)

    # ============= Pack =============

    async def export_ovpack(
        self,
        uri: str,
        to: str,
        include_vectors: bool = False,
        *,
        identity: OVIdentity | None = None,
    ) -> str:
        """Export context as .ovpack file and save to local path.

        Args:
            uri: Viking URI to export
            to: Local file path where to save the .ovpack file

        Returns:
            Local file path where the .ovpack was saved
        """
        uri = VikingURI.normalize(uri)

        # Determine target path
        to_path = Path(to)
        if to_path.is_dir():
            base_name = uri.strip().rstrip("/").split("/")[-1]
            if not base_name:
                base_name = "export"
            to_path = to_path / f"{base_name}.ovpack"
        elif not str(to_path).endswith(".ovpack"):
            to_path = Path(str(to_path) + ".ovpack")

        # Ensure parent directory exists
        to_path.parent.mkdir(parents=True, exist_ok=True)

        # Request export and stream response
        response = await self._request(
            "POST",
            "/api/v1/pack/export",
            json={"uri": uri, "include_vectors": include_vectors},
            **self._request_options(identity),
        )

        # Check for errors
        if not response.is_success:
            self._handle_response(response)

        # Save streamed content to local file
        with open(to_path, "wb") as f:
            f.write(response.content)

        return str(to_path)

    async def backup_ovpack(
        self, to: str, include_vectors: bool = False, *, identity: OVIdentity | None = None
    ) -> str:
        """Back up public scopes as a restore-only .ovpack file."""
        to_path = Path(to)
        if to_path.is_dir():
            to_path = to_path / "openviking-backup.ovpack"
        elif not str(to_path).endswith(".ovpack"):
            to_path = Path(str(to_path) + ".ovpack")

        to_path.parent.mkdir(parents=True, exist_ok=True)

        response = await self._request(
            "POST",
            "/api/v1/pack/backup",
            json={"include_vectors": include_vectors},
            **self._request_options(identity),
        )
        if not response.is_success:
            self._handle_response(response)

        with open(to_path, "wb") as f:
            f.write(response.content)

        return str(to_path)

    async def import_ovpack(
        self,
        file_path: str,
        parent: str,
        on_conflict: Optional[str] = None,
        vector_mode: Optional[str] = None,
        *,
        identity: OVIdentity | None = None,
    ) -> str:
        """Import .ovpack file."""
        parent = VikingURI.normalize(parent)
        request_data = {
            "parent": parent,
        }
        if on_conflict is not None:
            request_data["on_conflict"] = on_conflict
        if vector_mode is not None:
            request_data["vector_mode"] = vector_mode

        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Local ovpack file not found: {file_path}")
        if not file_path_obj.is_file():
            raise ValueError(f"Path {file_path} is not a file")

        temp_file_id = await self._upload_temp_file(file_path, **_identity_arguments(identity))
        request_data["temp_file_id"] = temp_file_id

        response = await self._request(
            "POST", "/api/v1/pack/import", json=request_data, **self._request_options(identity)
        )
        result = self._handle_response(response)
        return result.get("uri", "")

    async def restore_ovpack(
        self,
        file_path: str,
        on_conflict: Optional[str] = None,
        vector_mode: Optional[str] = None,
        *,
        identity: OVIdentity | None = None,
    ) -> str:
        """Restore backup .ovpack file."""
        request_data = {}
        if on_conflict is not None:
            request_data["on_conflict"] = on_conflict
        if vector_mode is not None:
            request_data["vector_mode"] = vector_mode

        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Local ovpack file not found: {file_path}")
        if not file_path_obj.is_file():
            raise ValueError(f"Path {file_path} is not a file")

        temp_file_id = await self._upload_temp_file(file_path, **_identity_arguments(identity))
        request_data["temp_file_id"] = temp_file_id

        response = await self._request(
            "POST", "/api/v1/pack/restore", json=request_data, **self._request_options(identity)
        )
        result = self._handle_response(response)
        return result.get("uri", "")

    # ============= Debug =============

    async def check_consistency(
        self, uri: str, *, identity: OVIdentity | None = None
    ) -> Dict[str, Any]:
        """Check filesystem/vector-index consistency for a URI subtree."""
        response = await self._request(
            "POST",
            "/api/v1/system/consistency",
            json={"uri": VikingURI.normalize(uri)},
            **self._request_options(identity),
        )
        return self._handle_response(response)

    async def health(self) -> bool:
        """Check server health."""
        try:
            response = await self._request("GET", "/health")
            data = response.json()
            return data.get("status") == "ok"
        except Exception:
            return False

    async def reindex(
        self,
        uri: str,
        mode: str = "vectors_only",
        wait: bool = True,
        *,
        identity: OVIdentity | None = None,
    ) -> Dict[str, Any]:
        """Trigger reindex for a URI."""
        response = await self._request(
            "POST",
            "/api/v1/content/reindex",
            json={
                "uri": uri,
                "mode": mode,
                "wait": wait,
            },
            **self._request_options(identity),
        )
        return self._handle_response(response)

    # ============= Observer (Internal) =============

    async def _get_queue_status(self) -> Dict[str, Any]:
        """Get queue system status (internal for _HTTPObserver)."""
        response = await self._request("GET", "/api/v1/observer/queue")
        return self._handle_response(response)

    async def _get_vikingdb_status(self) -> Dict[str, Any]:
        """Get VikingDB status (internal for _HTTPObserver)."""
        response = await self._request("GET", "/api/v1/observer/vikingdb")
        return self._handle_response(response)

    async def _get_models_status(self) -> Dict[str, Any]:
        """Get models status (VLM, Embedding, Rerank) (internal for _HTTPObserver)."""
        response = await self._request("GET", "/api/v1/observer/models")
        return self._handle_response(response)

    async def _get_system_status(self) -> Dict[str, Any]:
        """Get system overall status (internal for _HTTPObserver)."""
        response = await self._request("GET", "/api/v1/observer/system")
        return self._handle_response(response)

    # ============= Admin =============

    async def admin_create_account(self, account_id: str, admin_user_id: str) -> Dict[str, Any]:
        """Create a new account with its first admin user."""
        response = await self._request(
            "POST",
            "/api/v1/admin/accounts",
            json={"account_id": account_id, "admin_user_id": admin_user_id},
        )
        return self._handle_response(response)

    async def admin_list_accounts(self) -> List[Any]:
        """List all accounts."""
        response = await self._request("GET", "/api/v1/admin/accounts")
        return self._handle_response(response)

    async def admin_delete_account(self, account_id: str) -> Dict[str, Any]:
        """Delete an account and all associated users."""
        response = await self._request("DELETE", f"/api/v1/admin/accounts/{account_id}")
        return self._handle_response(response)

    async def admin_register_user(
        self, account_id: str, user_id: str, role: str = "user"
    ) -> Dict[str, Any]:
        """Register a new user in an account."""
        response = await self._request(
            "POST",
            f"/api/v1/admin/accounts/{account_id}/users",
            json={"user_id": user_id, "role": role},
        )
        return self._handle_response(response)

    async def admin_list_users(self, account_id: str) -> List[Any]:
        """List all users in an account."""
        response = await self._request("GET", f"/api/v1/admin/accounts/{account_id}/users")
        return self._handle_response(response)

    async def admin_remove_user(self, account_id: str, user_id: str) -> Dict[str, Any]:
        """Remove a user from an account."""
        response = await self._request(
            "DELETE", f"/api/v1/admin/accounts/{account_id}/users/{user_id}"
        )
        return self._handle_response(response)

    async def admin_set_role(self, account_id: str, user_id: str, role: str) -> Dict[str, Any]:
        """Change a user's role."""
        response = await self._request(
            "PUT",
            f"/api/v1/admin/accounts/{account_id}/users/{user_id}/role",
            json={"role": role},
        )
        return self._handle_response(response)

    async def admin_regenerate_key(self, account_id: str, user_id: str) -> Dict[str, Any]:
        """Regenerate a user's API key. Old key is immediately invalidated."""
        response = await self._request(
            "POST",
            f"/api/v1/admin/accounts/{account_id}/users/{user_id}/key",
        )
        return self._handle_response(response)

    async def admin_migrate(self, cleanup: bool = False) -> Dict[str, Any]:
        """Start legacy data migration or legacy namespace cleanup."""
        action = "cleanup" if cleanup else "migrate"
        response = await self._request("POST", "/api/v1/admin/migrate", json={"action": action})
        return self._handle_response(response)

    # ============= New methods for BaseClient interface =============

    def session(self, session_id: Optional[str] = None, must_exist: bool = False) -> Any:
        """Create a new session or load an existing one.

        Args:
            session_id: Session ID, creates a new session if None
            must_exist: If True and session_id is provided, raises NotFoundError
                        when the session does not exist.
                        If session_id is None, must_exist is ignored.

        Returns:
            Session object

        Raises:
            NotFoundError: If must_exist=True and the session does not exist.
        """
        from openviking.client.session import Session

        if not session_id:
            result = run_async(self.create_session())
            session_id = result.get("session_id", "")
        elif must_exist:
            # get_session() raises NotFoundError (via _handle_response) for 404.
            run_async(self.get_session(session_id))
        return Session(self, session_id, self._user)

    async def session_exists(self, session_id: str, *, identity: OVIdentity | None = None) -> bool:
        """Check whether a session exists in storage.

        Args:
            session_id: Session ID to check

        Returns:
            True if the session exists, False otherwise
        """
        try:
            await self.get_session(session_id, identity=identity)
            return True
        except NotFoundError:
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get system status.

        Returns:
            Dict containing health status of all components.
        """
        return self._observer.system

    def is_healthy(self) -> bool:
        """Quick health check (synchronous).

        Returns:
            True if all components are healthy, False otherwise.
        """
        return self._observer.is_healthy()

    @property
    def observer(self) -> _HTTPObserver:
        """Get observer service for component status."""
        return self._observer
