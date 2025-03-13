"""
LightRAG FastAPI Server with Flamingo Integration
"""

from fastapi import FastAPI, Depends, HTTPException, status
import asyncio
import os
import logging
import logging.config
import uvicorn
import pipmaster as pm
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import configparser
from ascii_colors import ASCIIColors
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from lightrag.api.utils_api import (
    get_api_key_dependency,
    parse_args,
    display_splash_screen,
)
from lightrag import LightRAG
from lightrag.api import __api_version__
from lightrag.utils import EmbeddingFunc
from lightrag.api.routers.document_routes import (
    DocumentManager,
    create_document_routes,
    run_scanning_process,
)
from lightrag.api.routers.query_routes import create_query_routes
from lightrag.api.routers.graph_routes import create_graph_routes

from lightrag.utils import logger, set_verbose_debug
from lightrag.kg.shared_storage import (
    get_namespace_data,
    get_pipeline_status_lock,
    initialize_pipeline_status,
    get_all_update_flags_status,
)
from fastapi.security import OAuth2PasswordRequestForm
from .auth import auth_handler

# Import Flamingo-specific modules
from flamingo import flamingo_complete_if_cache
import numpy as np
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv(".env", override=True)

# Initialize config parser
config = configparser.ConfigParser()
config.read("config.ini")


class LightragPathFilter(logging.Filter):
    """Filter for lightrag logger to filter out frequent path access logs"""

    def __init__(self):
        super().__init__()
        # Define paths to be filtered
        self.filtered_paths = ["/documents", "/health", "/webui/"]

    def filter(self, record):
        try:
            # Check if record has the required attributes for an access log
            if not hasattr(record, "args") or not isinstance(record.args, tuple):
                return True
            if len(record.args) < 5:
                return True

            # Extract method, path and status from the record args
            method = record.args[1]
            path = record.args[2]
            status = record.args[4]

            # Filter out successful GET requests to filtered paths
            if (
                method == "GET"
                and (status == 200 or status == 304)
                and path in self.filtered_paths
            ):
                return False

            return True
        except Exception:
            # In case of any error, let the message through
            return True


# Flamingo implementation for LLM
async def flamingo_llm_model_func(
    prompt, system_prompt=None, history_messages=None, keyword_extraction=False, **kwargs
) -> str:
    if history_messages is None:
        history_messages = []
    
    return await flamingo_complete_if_cache(
        model=os.getenv("FLAMINGO_MODEL", "llama3"),
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        subscription_id=os.getenv("SUBSCRIPTION_ID"),
        base_url=os.getenv("BASE_URL"),
        client_id=os.getenv("CLIENT_ID"),
        client_secret=os.getenv("CLIENT_SECRET"),
        subscription_key=os.getenv("SUBSCRIPTION_KEY"),
        tenant_id=os.getenv("TENANT_ID"),
        **kwargs,
    )


# Initialize sentence transformer model once at module level
_sentence_transformer_model = None


def get_sentence_transformer():
    global _sentence_transformer_model
    if _sentence_transformer_model is None:
        _sentence_transformer_model = SentenceTransformer(
            os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-mpnet-base-v2")
        )
    return _sentence_transformer_model


# Embedding function using SentenceTransformer
async def embedding_func(texts: list[str]) -> np.ndarray:
    model = get_sentence_transformer()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings


async def get_embedding_dimension():
    """Determine embedding dimension by encoding a test sentence"""
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    return embedding.shape[1]


def create_app(args):
    # Setup logging
    logger.setLevel(args.log_level)
    set_verbose_debug(args.verbose)

    # Check if API key is provided either through env var or args
    api_key = os.getenv("LIGHTRAG_API_KEY") or args.key

    # Initialize document manager
    doc_manager = DocumentManager(args.input_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for startup and shutdown events"""
        # Store background tasks
        app.state.background_tasks = set()

        try:
            # Initialize database connections
            await rag.initialize_storages()

            await initialize_pipeline_status()
            pipeline_status = await get_namespace_data("pipeline_status")

            should_start_autoscan = False
            async with get_pipeline_status_lock():
                # Auto scan documents if enabled
                if args.auto_scan_at_startup:
                    if not pipeline_status.get("autoscanned", False):
                        pipeline_status["autoscanned"] = True
                        should_start_autoscan = True

            # Only run auto scan when no other process started it first
            if should_start_autoscan:
                # Create background task
                task = asyncio.create_task(run_scanning_process(rag, doc_manager))
                app.state.background_tasks.add(task)
                task.add_done_callback(app.state.background_tasks.discard)
                logger.info(f"Process {os.getpid()} auto scan task started at startup.")

            ASCIIColors.green("\nServer is ready to accept connections! 🚀\n")

            yield

        finally:
            # Clean up database connections
            await rag.finalize_storages()

    # Initialize FastAPI
    app = FastAPI(
        title="LightRAG API with Flamingo",
        description="API for querying text using LightRAG with Flamingo LLM integration"
        + "(With authentication)"
        if api_key
        else "",
        version=__api_version__,
        openapi_tags=[{"name": "api"}],
        lifespan=lifespan,
    )

    def get_cors_origins():
        """Get allowed origins from environment variable"""
        origins_str = os.getenv("CORS_ORIGINS", "*")
        if origins_str == "*":
            return ["*"]
        return [origin.strip() for origin in origins_str.split(",")]

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Create the optional API key dependency
    optional_api_key = get_api_key_dependency(api_key)

    # Create working directory if it doesn't exist
    Path(args.working_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize LightRAG with Flamingo and SentenceTransformer
    embedding_dim = asyncio.run(get_embedding_dimension())
    logger.info(f"Using embedding dimension: {embedding_dim}")
    
    global rag
    rag = LightRAG(
        working_dir=args.working_dir,
        llm_model_func=flamingo_llm_model_func,
        llm_model_name=os.getenv("FLAMINGO_MODEL", "llama3"),
        llm_model_max_async=args.max_async,
        llm_model_max_token_size=args.max_tokens,
        chunk_token_size=int(args.chunk_size),
        chunk_overlap_token_size=int(args.chunk_overlap_size),
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=args.max_embed_tokens,
            func=embedding_func,
        ),
        kv_storage=args.kv_storage,
        graph_storage=args.graph_storage,
        vector_storage=args.vector_storage,
        doc_status_storage=args.doc_status_storage,
        vector_db_storage_cls_kwargs={
            "cosine_better_than_threshold": args.cosine_threshold
        },
        enable_llm_cache_for_entity_extract=args.enable_llm_cache_for_extract,
        embedding_cache_config={
            "enabled": True,
            "similarity_threshold": 0.95,
            "use_llm_check": False,
        },
        namespace_prefix=args.namespace_prefix,
        auto_manage_storages_states=False,
    )

    # Add routes
    app.include_router(create_document_routes(rag, doc_manager, api_key))
    app.include_router(create_query_routes(rag, api_key, args.top_k))
    app.include_router(create_graph_routes(rag, api_key))

    @app.post("/login")
    async def login(form_data: OAuth2PasswordRequestForm = Depends()):
        username = os.getenv("AUTH_USERNAME")
        password = os.getenv("AUTH_PASSWORD")

        if not (username and password):
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Authentication not configured",
            )

        if form_data.username != username or form_data.password != password:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect credentials"
            )

        return {
            "access_token": auth_handler.create_token(username),
            "token_type": "bearer",
        }

    @app.get("/health", dependencies=[Depends(optional_api_key)])
    async def get_status():
        """Get current system status"""
        # Get update flags status for all namespaces
        update_status = await get_all_update_flags_status()

        return {
            "status": "healthy",
            "working_directory": str(args.working_dir),
            "input_directory": str(args.input_dir),
            "configuration": {
                "llm_model": os.getenv("FLAMINGO_MODEL", "llama3"),
                "embedding_model": os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"),
                "max_tokens": args.max_tokens,
                "kv_storage": args.kv_storage,
                "doc_status_storage": args.doc_status_storage,
                "graph_storage": args.graph_storage,
                "vector_storage": args.vector_storage,
                "enable_llm_cache_for_extract": args.enable_llm_cache_for_extract,
            },
            "update_status": update_status,
        }

    # Webui mount webui/index.html
    static_dir = Path(__file__).parent / "webui"
    static_dir.mkdir(exist_ok=True)
    app.mount(
        "/webui",
        StaticFiles(directory=static_dir, html=True, check_dir=True),
        name="webui",
    )

    return app


def get_application(args=None):
    """Factory function for creating the FastAPI application"""
    if args is None:
        args = parse_args()
    return create_app(args)


def configure_logging():
    """Configure logging for uvicorn startup"""
    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.filters = []

    # Get log directory path from environment variable
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "lightrag.log"))

    print(f"\nLightRAG log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "uvicorn": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
                "uvicorn.access": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                    "filters": ["path_filter"],
                },
                "uvicorn.error": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                    "filters": ["path_filter"],
                },
            },
            "filters": {
                "path_filter": {
                    "()": "lightrag.api.lightrag_server.LightragPathFilter",
                },
            },
        }
    )


def check_and_install_dependencies():
    """Check and install required dependencies"""
    required_packages = [
        "uvicorn",
        "tiktoken",
        "fastapi",
        "sentence-transformers",
        # Add other required packages here
    ]

    for package in required_packages:
        if not pm.is_installed(package):
            print(f"Installing {package}...")
            pm.install(package)
            print(f"{package} installed successfully")


def main():
    # Check if running under Gunicorn
    if "GUNICORN_CMD_ARGS" in os.environ:
        # If started with Gunicorn, return directly as Gunicorn will call get_application
        print("Running under Gunicorn - worker management handled by Gunicorn")
        return

    # Check and install dependencies
    check_and_install_dependencies()

    from multiprocessing import freeze_support

    freeze_support()

    # Configure logging before parsing args
    configure_logging()

    args = parse_args(is_uvicorn_mode=True)
    display_splash_screen(args)

    # Create application instance directly instead of using factory function
    app = create_app(args)

    # Start Uvicorn in single process mode
    uvicorn_config = {
        "app": app,  # Pass application instance directly instead of string path
        "host": args.host,
        "port": args.port,
        "log_config": None,  # Disable default config
    }

    if args.ssl:
        uvicorn_config.update(
            {
                "ssl_certfile": args.ssl_certfile,
                "ssl_keyfile": args.ssl_keyfile,
            }
        )

    print(f"Starting Uvicorn server in single-process mode on {args.host}:{args.port}")
    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    main()
