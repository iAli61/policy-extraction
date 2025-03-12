# lightrag README

# LightRAG

LightRAG is a Python package designed for integrating with various language models, including OpenAI and Flamingo APIs. This package provides a simple and efficient way to interact with these models for tasks such as text generation, embeddings, and more.

## Installation

To install the required packages, run the following commands:

```bash
pip install openai==1.30.1
pip install msal==1.28.0
pip install httpx==0.27.0
```

## Usage
installation:
```
pip install -e "git+https://github.com/iAli61/lightrag.git@b80a8bb93618801d66a061e2e143734a82a48a90#egg=lightrag-hku[api]"
```

### OpenAI Integration

To use the OpenAI API, you can refer to the `lightrag_openai_compatible_demo.py` file for examples on how to interact with the API. The `openai.py` file contains the necessary client implementation.

### Flamingo Integration

For integrating with the Flamingo API, you can check the `lightrag_flamingo_demo.py` file. The `flamingo.py` file provides the client implementation for making requests to the Flamingo service.

## Directory Structure

The project is organized as follows:

```
lightrag/
├── __init__.py
├── flamingo.py
├── flamingo_client.py
├── lightrag_flamingo_demo.py
├── lightrag_openai_compatible_demo.py
├── openai.py
├── utils/
│   ├── __init__.py
│   └── helpers.py
├── kg/
│   ├── __init__.py 
│   └── shared_storage.py
└── README.md
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.