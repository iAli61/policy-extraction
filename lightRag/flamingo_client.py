### please re-run make install and make sure you have following versions of required packages
# !pip install openai==1.30.1
# !pip install msal==1.28.0
# !pip install httpx==0.27.0

from __future__ import annotations

import datetime
from enum import Enum

import msal
from openai import OpenAI


def get_error(msg):
    def raise_not_implemented_error(*args, **kwargs):
        raise NotImplementedError(msg)

    return raise_not_implemented_error


def create_error(obj, msg, method_names):
    for method in method_names:
        setattr(obj, method, get_error(msg))
    return obj


class AuthException(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class FlamingoLLMClient(OpenAI):

    def __init__(self, subscription_id: str, base_url: str, client_id: str, client_secret: str, subscription_key: str, tenant: str):
        super().__init__(base_url=self._format_url(base_url), api_key="")
        self.subscription_id = subscription_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant = tenant
        if subscription_key is None:
            raise TypeError("subscription_key cannot be None")
        self.subscription_key = subscription_key
        self._token = None
        self._token_expiry = None
        self.embeddings = create_error(
            self.embeddings, msg="model has no embeddings endpoint! use default service instead", method_names=("create",)
        )
        self.files = create_error(
            self.files,
            msg="model has no files endpoint!",
            method_names=("create", "list", "delete", "retrieve", "content", "retrieve_content"),
        )
        self.images = create_error(
            self.images, msg="model has no images endpoint!", method_names=("create_variation", "generate", "edit")
        )
        self.audio = create_error(
            self.audio, msg="model has no audio endpoint!", method_names=("speech", "transcriptions", "translations")
        )
        self.moderations = create_error(self.moderations, msg="model has no moderations endpoint!", method_names=("create",))
        self.models = create_error(
            self.models, msg="model has no models endpoint!", method_names=("delete", "retrieve", "list")
        )
        self.fine_tuning = create_error(self.fine_tuning, msg="model has no fine_tuning endpoint!", method_names=("jobs",))
        self.beta = create_error(
            self.beta, msg="model has no beta endpoint!", method_names=("threads", "assistants", "vector_stores")
        )
        self.batches = create_error(
            self.batches, msg="model has no batches endpoint!", method_names=("create", "list", "retrieve", "cancel")
        )

    @staticmethod
    def _format_url(url: str) -> str:
        if not (url.endswith("/v1") or url.endswith("/v1/")):
            if url.endswith("/"):
                return f"{url}v1"
            else:
                return f"{url}/v1"
        return url

    @property
    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._get_token()}", "Ocp-Apim-Subscription-Key": self.subscription_key}

    def _get_token(self) -> str:
        if self._token is None or self._token_expiry <= datetime.datetime.now():
            app = msal.ConfidentialClientApplication(
                client_id=self.client_id,
                client_credential=self.client_secret,
                authority=f"https://login.microsoftonline.com/{self.tenant}",
            )
            response = app.acquire_token_for_client(scopes=[f"{self.subscription_id}/.default"])
            if "access_token" in response:
                self._token = response["access_token"]
                self._token_expiry = datetime.datetime.now() + datetime.timedelta(seconds=response["expires_in"])
                return self._token
            else:
                raise AuthException(f"could not authorize. Response was {response}")
        else:
            
            return self._token
