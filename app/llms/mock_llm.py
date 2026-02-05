"""
Comprehensive Mock LLM - Compatible with BaseLLM interface
Covers React, FastAPI, and AI/LLM questions from beginner to advanced levels
"""

from app.core.llm_interface import BaseLLM


class MockLLM(BaseLLM):
    """
    Mock LLM with comprehensive knowledge base covering:
    - React (30+ topics)
    - FastAPI (30+ topics)
    - AI/LLM (30+ topics)
    From beginner to advanced levels
    """
    
    def __init__(self):
        super().__init__()
        self.knowledge_base = self._build_knowledge_base()
    
    def generate(self, prompt: str) -> str:
        """
        Generate response based on prompt by searching knowledge base
        
        Args:
            prompt: User query/prompt
            
        Returns:
            Relevant response from knowledge base
        """
        p = prompt.lower().strip()
        
        # Search through all knowledge for best match
        response = self._find_best_match(p)
        
        if response:
            return response
        
        # Fallback response
        return "This is a mock AI response. I can answer questions about React, FastAPI, and AI/LLM topics from beginner to advanced levels. Try asking about hooks, authentication, transformers, or any related topic!"
    
    def _find_best_match(self, query: str) -> str:
        """Find best matching response from knowledge base"""
        # Try exact/partial keyword matches first
        for key, value in self.knowledge_base.items():
            if key in query or self._fuzzy_match(key, query):
                return value
        
        # Try multi-word matches
        for key, value in self.knowledge_base.items():
            key_words = key.split()
            if len(key_words) > 1 and all(word in query for word in key_words):
                return value
        
        return None
    
    def _fuzzy_match(self, key: str, query: str) -> bool:
        """Check if key words appear in query"""
        key_words = key.split()
        return any(word in query for word in key_words if len(word) > 3)
    
    def _build_knowledge_base(self) -> dict:
        """Build comprehensive knowledge base"""
        return {
            # ==================== REACT ====================
            # BEGINNER
            'what is react': 'React is a JavaScript library for building user interfaces, developed by Facebook. It allows developers to create reusable UI components and efficiently update the DOM using a virtual DOM.',
            
            'react component': 'React components are the building blocks of a React application. They can be function components (recommended) or class components, and they accept props and return React elements that describe what should appear on the screen.',
            
            'jsx': 'JSX (JavaScript XML) is a syntax extension for JavaScript that allows you to write HTML-like code in your JavaScript files. It gets transpiled to React.createElement() calls by tools like Babel.',
            
            'what are props': 'Props (properties) are read-only data passed from parent to child components. They enable component reusability and allow data to flow down the component tree in a unidirectional manner.',
            
            'react state': 'State is a built-in object that stores component data that can change over time. When state changes, the component re-renders to reflect the new data. Use useState hook in function components.',
            
            'virtual dom': 'The Virtual DOM is a lightweight copy of the actual DOM. React uses it to optimize updates by calculating the minimal changes needed (diffing) and batch updating the real DOM for better performance.',
            
            'react render': 'Rendering in React is the process of converting components into DOM elements. React calls the render method/function, creates a virtual DOM, compares it with the previous version, and updates only changed parts.',
            
            # INTERMEDIATE
            'react hooks': 'React Hooks are functions that let you use state and lifecycle features in function components. Common hooks include useState, useEffect, useContext, useReducer, useMemo, useCallback, and useRef. Introduced in React 16.8.',
            
            'usestate': 'useState is a Hook that adds state to function components. Syntax: const [state, setState] = useState(initialValue). It returns current state and a function to update it. Triggers re-render when state changes.',
            
            'useeffect': 'useEffect is a Hook for side effects in function components. It runs after render and can optionally clean up. It combines componentDidMount, componentDidUpdate, and componentWillUnmount. Syntax: useEffect(() => { /* effect */ }, [dependencies]).',
            
            'usecontext': 'useContext is a Hook that allows you to consume context values without wrapping components in Context.Consumer. It provides a way to pass data through the component tree without manual prop drilling. Usage: const value = useContext(MyContext).',
            
            'usereducer': 'useReducer is a Hook for managing complex state logic. It\'s an alternative to useState that works like Redux reducers. Syntax: const [state, dispatch] = useReducer(reducer, initialState). Better for state with multiple sub-values or complex transitions.',
            
            'react lifecycle': 'React component lifecycle has three phases: Mounting (constructor, render, componentDidMount), Updating (render, componentDidUpdate), and Unmounting (componentWillUnmount). In function components, useEffect handles all lifecycle events.',
            
            'controlled component': 'Controlled components are form elements whose values are controlled by React state. The component state becomes the "single source of truth". Input changes trigger state updates via onChange handlers.',
            
            'uncontrolled component': 'Uncontrolled components store their own state in the DOM. You access values using refs instead of state. Useful for simple forms or integrating with non-React code.',
            
            'lifting state up': 'Lifting state up is moving state to the closest common ancestor component when multiple components need to share the same changing data. Enables data flow between sibling components.',
            
            'react router': 'React Router is a standard library for routing in React applications. It provides components like BrowserRouter, Route, Routes, Link, and Navigate for client-side navigation without page reloads.',
            
            'react forms': 'React forms use controlled components to manage form data. Handle input changes with onChange, form submission with onSubmit. Validate data in state before submission. Libraries like Formik and React Hook Form simplify complex forms.',
            
            # ADVANCED
            'usememo': 'useMemo is a Hook that memoizes expensive calculations, only recalculating when dependencies change. Syntax: const memoizedValue = useMemo(() => computeExpensiveValue(a, b), [a, b]). Optimizes performance by avoiding unnecessary computations.',
            
            'usecallback': 'useCallback is a Hook that memoizes callback functions to prevent unnecessary re-renders of child components. Returns a memoized version that only changes if dependencies change. Syntax: const memoizedCallback = useCallback(() => { doSomething(a, b) }, [a, b]).',
            
            'useref': 'useRef is a Hook that creates a mutable ref object persisting across renders. Used for accessing DOM elements directly or storing mutable values without triggering re-renders. Syntax: const refContainer = useRef(initialValue).',
            
            'custom hooks': 'Custom Hooks are JavaScript functions starting with "use" that can call other Hooks. They enable reusing stateful logic across multiple components without changing component hierarchy. Extract common patterns into reusable functions.',
            
            'react context api': 'Context API provides a way to pass data through the component tree without prop drilling. Includes createContext, Provider component, and Consumer/useContext for global state management. Good for themes, auth, language.',
            
            'react performance': 'React performance optimization includes: React.memo for component memoization, useMemo/useCallback for value/function memoization, code splitting with lazy/Suspense, virtualization for long lists, avoiding inline functions, proper key usage.',
            
            'react memo': 'React.memo is a higher-order component that memoizes functional components. It prevents re-renders if props haven\'t changed (shallow comparison). Similar to PureComponent for class components. Syntax: const MemoizedComponent = React.memo(MyComponent).',
            
            'code splitting': 'Code splitting in React uses dynamic import() and React.lazy() to split code into smaller chunks that load on demand, improving initial load time. Used with Suspense for loading states. Example: const Component = lazy(() => import(\'./Component\')).',
            
            'error boundaries': 'Error boundaries are React components that catch JavaScript errors in their child component tree, log errors, and display fallback UI. Use componentDidCatch and static getDerivedStateFromError. Functional components don\'t support error boundaries yet.',
            
            'react portals': 'React Portals provide a way to render children into a DOM node outside the parent component\'s DOM hierarchy. Useful for modals, tooltips, dropdowns. Syntax: ReactDOM.createPortal(child, container).',
            
            'render props': 'Render props is a pattern where a component takes a function as a prop and calls it to determine what to render. Enables code reuse and component composition. Example: <DataProvider render={data => <div>{data}</div>} />.',
            
            'higher order component': 'Higher-Order Components (HOC) are functions that take a component and return a new component with additional props or behavior. Enable component logic reuse. Pattern: const EnhancedComponent = higherOrderComponent(WrappedComponent).',
            
            'react fiber': 'React Fiber is the reconciliation algorithm enabling incremental rendering. It can pause, prioritize, and resume rendering work, making apps more responsive. Introduced in React 16 for better handling of animations, layout, and gestures.',
            
            'concurrent mode': 'Concurrent Features (formerly Concurrent Mode) allow React to interrupt rendering to handle high-priority updates. Features include useTransition for non-urgent updates, useDeferredValue for deferring expensive renders, and Suspense for data fetching.',
            
            'react server components': 'React Server Components render on the server, reducing bundle size and enabling direct database access. They run only on the server and don\'t ship JavaScript to the client. Can mix with client components for optimal performance.',
            
            'suspense': 'Suspense is a component that displays a fallback while waiting for lazy-loaded components or data to load. Works with React.lazy() and frameworks like Next.js for data fetching. Syntax: <Suspense fallback={<Loading />}><LazyComponent /></Suspense>.',
            
            'redux': 'Redux is a predictable state container for JavaScript apps. Uses a single store, actions (what happened), and reducers (how state changes). React-Redux connects components to store with useSelector and useDispatch hooks.',
            
            'react testing': 'React testing uses Jest and React Testing Library. Best practices: test user behavior over implementation, use accessible queries (getByRole, getByLabelText), avoid testing implementation details, mock external dependencies.',
            
            'react native': 'React Native is a framework for building native mobile apps using React. Uses native components (View, Text) instead of web components (div, span). Enables code sharing between iOS and Android with platform-specific code when needed.',
            
            'react keys': 'Keys help React identify which items have changed, added, or removed in lists. Use stable, unique IDs (not array indexes). Keys should be unique among siblings. Example: items.map(item => <li key={item.id}>{item.name}</li>).',
            
            'prop drilling': 'Prop drilling is passing props through multiple component levels to reach deeply nested components. Can be avoided using Context API, Redux, or component composition patterns.',
            
            # ==================== FASTAPI ====================
            # BEGINNER
            'what is fastapi': 'FastAPI is a modern, fast (high-performance) Python web framework for building APIs with Python 3.7+ based on standard Python type hints. Built on Starlette for web parts and Pydantic for data validation.',
            
            'fastapi vs flask': 'FastAPI is faster than Flask (comparable to Node.js and Go), has automatic API documentation (Swagger/OpenAPI), built-in data validation with Pydantic, native async support, and automatic JSON serialization. Flask is more mature with larger ecosystem.',
            
            'install fastapi': 'Install FastAPI with: pip install fastapi uvicorn[standard]. FastAPI is the framework, uvicorn is the ASGI server needed to run FastAPI applications. For production, also install gunicorn.',
            
            'first fastapi app': 'Create basic FastAPI app:\nfrom fastapi import FastAPI\napp = FastAPI()\n\n@app.get("/")\ndef read_root():\n    return {"Hello": "World"}\n\nRun with: uvicorn main:app --reload (--reload for development auto-restart).',
            
            'path parameters': 'Path parameters are variables in the URL path. Example: @app.get("/items/{item_id}") def read_item(item_id: int). FastAPI automatically validates and converts types based on type hints.',
            
            'query parameters': 'Query parameters are optional parameters after ? in URLs. Define as function parameters with default values: @app.get("/items/") def read_items(skip: int = 0, limit: int = 10). FastAPI validates types automatically.',
            
            'request body': 'Request bodies in FastAPI use Pydantic models:\nfrom pydantic import BaseModel\nclass Item(BaseModel):\n    name: str\n    price: float\n\n@app.post("/items/")\ndef create_item(item: Item):\n    return item',
            
            'fastapi response': 'FastAPI responses are automatically converted to JSON. Return dicts, lists, Pydantic models, or use Response/JSONResponse for custom responses. Set status codes and headers easily.',
            
            # INTERMEDIATE
            'pydantic models': 'Pydantic models provide automatic data validation, serialization, and documentation. Use Python type hints for validation. Support nested models, custom validators with @validator, and field constraints with Field().',
            
            'dependency injection': 'FastAPI\'s dependency injection uses Depends() to declare dependencies automatically resolved and injected into path operations. Enables code reuse, testing, security, database sessions. Example: def get_db(): ... then use db: Session = Depends(get_db).',
            
            'async await': 'FastAPI supports async/await for asynchronous operations. Use async def for I/O-bound operations (database queries, API calls, file operations). For CPU-bound tasks, use regular def functions which run in thread pool.',
            
            'background tasks': 'BackgroundTasks execute functions after returning response. Useful for emails, notifications, logging:\nfrom fastapi import BackgroundTasks\n@app.post("/send-email/")\nasync def send_email(background_tasks: BackgroundTasks):\n    background_tasks.add_task(send_email_task, email)',
            
            'fastapi validation': 'FastAPI provides automatic validation using Pydantic. Use Field() for additional constraints:\nfrom pydantic import Field\nname: str = Field(..., min_length=3, max_length=50)\nage: int = Field(..., gt=0, le=120)',
            
            'response model': 'Response models specify return type and filter/validate responses:\n@app.get("/items/", response_model=List[Item])\nEnsures only specified fields are returned. Use response_model_exclude_unset to exclude unset fields.',
            
            'status codes': 'FastAPI uses standard HTTP status codes from fastapi.status:\n@app.post("/items/", status_code=status.HTTP_201_CREATED)\nCommon: 200 OK, 201 Created, 204 No Content, 400 Bad Request, 404 Not Found, 500 Internal Server Error.',
            
            'file upload': 'Handle file uploads with File() and UploadFile:\nfrom fastapi import File, UploadFile\n@app.post("/uploadfile/")\nasync def create_upload_file(file: UploadFile = File(...)):\n    contents = await file.read()\n    return {"filename": file.filename}',
            
            'cors fastapi': 'Enable CORS with CORSMiddleware:\nfrom fastapi.middleware.cors import CORSMiddleware\napp.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])',
            
            'fastapi forms': 'Handle form data with Form():\nfrom fastapi import Form\n@app.post("/login/")\nasync def login(username: str = Form(...), password: str = Form(...)):\n    return {"username": username}',
            
            # ADVANCED
            'fastapi security': 'FastAPI provides security utilities for OAuth2, JWT tokens, API keys, HTTP Basic/Digest auth. Use OAuth2PasswordBearer for token-based auth:\nfrom fastapi.security import OAuth2PasswordBearer\noauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")',
            
            'jwt authentication': 'Implement JWT with python-jose:\nfrom jose import JWTError, jwt\nCreate tokens in login endpoint, verify with dependencies:\ndef get_current_user(token: str = Depends(oauth2_scheme)):\n    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])',
            
            'database integration': 'Integrate databases with SQLAlchemy:\nfrom sqlalchemy.orm import Session\ndef get_db():\n    db = SessionLocal()\n    try:\n        yield db\n    finally:\n        db.close()\n\n@app.get("/users/")\ndef read_users(db: Session = Depends(get_db)):\n    return db.query(User).all()',
            
            'fastapi middleware': 'Create custom middleware to process all requests/responses:\n@app.middleware("http")\nasync def add_process_time_header(request: Request, call_next):\n    start_time = time.time()\n    response = await call_next(request)\n    process_time = time.time() - start_time\n    response.headers["X-Process-Time"] = str(process_time)\n    return response',
            
            'websockets': 'FastAPI supports WebSockets for real-time communication:\nfrom fastapi import WebSocket\n@app.websocket("/ws")\nasync def websocket_endpoint(websocket: WebSocket):\n    await websocket.accept()\n    while True:\n        data = await websocket.receive_text()\n        await websocket.send_text(f"Message: {data}")',
            
            'testing fastapi': 'Test FastAPI with TestClient:\nfrom fastapi.testclient import TestClient\nclient = TestClient(app)\ndef test_read_main():\n    response = client.get("/")\n    assert response.status_code == 200\n    assert response.json() == {"msg": "Hello"}',
            
            'api versioning': 'Implement API versioning with APIRouter prefixes:\nfrom fastapi import APIRouter\nrouter_v1 = APIRouter(prefix="/api/v1", tags=["v1"])\nrouter_v2 = APIRouter(prefix="/api/v2", tags=["v2"])\napp.include_router(router_v1)\napp.include_router(router_v2)',
            
            'error handling': 'Custom exception handlers:\nfrom fastapi import HTTPException\nfrom fastapi.responses import JSONResponse\n@app.exception_handler(HTTPException)\nasync def http_exception_handler(request, exc):\n    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})',
            
            'graphql fastapi': 'Integrate GraphQL with Strawberry:\nimport strawberry\nfrom strawberry.fastapi import GraphQLRouter\n@strawberry.type\nclass Query:\n    @strawberry.field\n    def hello(self) -> str:\n        return "Hello"\nschema = strawberry.Schema(query=Query)\napp.include_router(GraphQLRouter(schema), prefix="/graphql")',
            
            'caching': 'Implement caching with Redis:\nfrom fastapi_cache import FastAPICache\nfrom fastapi_cache.backends.redis import RedisBackend\nfrom fastapi_cache.decorator import cache\n@app.get("/items/")\n@cache(expire=60)\nasync def get_items():\n    return {"items": [...])',
            
            'rate limiting': 'Add rate limiting with slowapi:\nfrom slowapi import Limiter, _rate_limit_exceeded_handler\nfrom slowapi.util import get_remote_address\nlimiter = Limiter(key_func=get_remote_address)\napp.state.limiter = limiter\n@app.get("/items/")\n@limiter.limit("5/minute")\nasync def read_items(request: Request):\n    return {"items": []}',
            
            'docker deployment': 'Dockerize FastAPI:\nFROM python:3.9-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\nCMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]',
            
            'fastapi performance': 'Performance optimization: use async for I/O operations, implement caching (Redis), use database connection pooling, optimize Pydantic with orm_mode, use response_model_exclude_unset, enable gzip compression, use CDN for static files.',
            
            'openapi schema': 'Customize OpenAPI schema:\n@app.get("/items/", tags=["items"], summary="Get items", description="Retrieve paginated items", response_description="List of items")\nModify generated schema: def custom_openapi(): ... app.openapi = custom_openapi',
            
            'event handlers': 'Use startup/shutdown events for initialization:\n@app.on_event("startup")\nasync def startup_event():\n    # Initialize database pool, cache, load ML models\n    pass\n\n@app.on_event("shutdown")\nasync def shutdown_event():\n    # Close database connections, cleanup\n    pass',
            
            # ==================== AI & LLM ====================
            # BEGINNER
            'what is ai': 'AI (Artificial Intelligence) enables machines to mimic human intelligence. Includes machine learning, natural language processing, computer vision, and robotics to perform tasks requiring human-like cognition such as learning, reasoning, and problem-solving.',
            
            'machine learning': 'Machine Learning is a subset of AI where systems learn from data without explicit programming. Three types: supervised learning (labeled data), unsupervised learning (unlabeled data), and reinforcement learning (reward-based).',
            
            'deep learning': 'Deep Learning uses neural networks with multiple layers (deep networks) to learn hierarchical representations of data. Powers image recognition, speech recognition, NLP, and autonomous vehicles. Subset of machine learning.',
            
            'neural network': 'Neural networks are computing systems inspired by biological neurons. Consist of interconnected nodes (neurons) organized in layers: input layer, hidden layers, and output layer. Learn by adjusting weights between connections.',
            
            'what is llm': 'LLM (Large Language Model) is an AI model trained on vast amounts of text data to understand and generate human-like text. Examples: GPT-4, Claude, BERT, LLaMA. Based on transformer architecture with billions of parameters.',
            
            'natural language processing': 'NLP (Natural Language Processing) enables computers to understand, interpret, and generate human language. Tasks include translation, sentiment analysis, named entity recognition, question answering, and text generation.',
            
            'supervised learning': 'Supervised learning trains models on labeled data where input-output pairs are known. Model learns to map inputs to correct outputs. Examples: classification (spam detection), regression (price prediction).',
            
            'unsupervised learning': 'Unsupervised learning finds patterns in unlabeled data without predefined outputs. Examples: clustering (customer segmentation), dimensionality reduction (PCA), anomaly detection, association rules.',
            
            'reinforcement learning': 'Reinforcement learning trains agents to make decisions by rewarding desired behaviors and punishing undesired ones. Agent learns through trial and error. Used in game AI, robotics, recommendation systems.',
            
            # INTERMEDIATE
            'transformer': 'Transformers are neural network architectures using self-attention mechanisms to process sequential data in parallel. Replaced RNNs for NLP tasks. Introduced in "Attention is All You Need" (2017). Foundation of modern LLMs.',
            
            'attention mechanism': 'Attention mechanisms allow models to focus on relevant parts of input when processing sequences. Self-attention computes relationships between all positions. Enables handling long-range dependencies and parallel processing.',
            
            'tokenization': 'Tokenization splits text into smaller units (tokens) for model processing. Methods: word-level (entire words), subword (BPE, WordPiece - handles rare words), character-level. Modern LLMs use subword tokenization (e.g., GPT uses BPE).',
            
            'embeddings': 'Embeddings are dense vector representations of words/tokens capturing semantic meaning. Similar words have similar embeddings in vector space. Types: Word2Vec, GloVe, contextual embeddings (BERT, GPT). Dimension typically 256-4096.',
            
            'bert': 'BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language model understanding context bidirectionally. Uses masked language modeling. Excels at classification, NER, question answering. Encoder-only architecture.',
            
            'gpt': 'GPT (Generative Pre-trained Transformer) is an autoregressive language model generating text by predicting next tokens. Decoder-only architecture. GPT-3 has 175B parameters. GPT-4 is multimodal with enhanced reasoning.',
            
            'fine tuning': 'Fine-tuning adapts pre-trained models to specific tasks by training on task-specific data with smaller learning rates. Requires less data than training from scratch. Techniques: full fine-tuning, LoRA, adapter layers.',
            
            'prompt engineering': 'Prompt engineering designs effective inputs to LLMs for desired outputs. Techniques: clear instructions, few-shot examples, chain-of-thought reasoning, role prompting, formatting (JSON, XML), temperature/top-p tuning.',
            
            'few shot learning': 'Few-shot learning enables models to learn from few examples in the prompt. 0-shot (no examples, just instructions), 1-shot (one example), few-shot (2-10 examples). LLMs perform tasks without explicit training.',
            
            'rag': 'RAG (Retrieval-Augmented Generation) combines retrieval systems with LLMs. Retrieves relevant documents from knowledge base and uses them as context for generation. Reduces hallucinations, enables knowledge updates without retraining.',
            
            'transfer learning': 'Transfer learning uses knowledge from one task to improve performance on another. Pre-train on large dataset, fine-tune on specific task. Enables leveraging expensive pre-training for downstream tasks.',
            
            # ADVANCED
            'transformer architecture': 'Transformer architecture: encoder-decoder structure with multi-head self-attention, feed-forward networks, layer normalization, residual connections. Key: Q (query), K (key), V (value) matrices. Positional encodings for sequence order. Scaled dot-product attention.',
            
            'self attention': 'Self-attention computes attention scores between all sequence positions using Q, K, V matrices. Formula: Attention(Q,K,V) = softmax(QK^T/√d_k)V. Scaled by √d_k to prevent gradient issues. Multi-head attention uses multiple parallel attention layers.',
            
            'positional encoding': 'Positional encodings add position information to token embeddings since transformers don\'t have inherent sequence order. Sinusoidal functions: PE(pos,2i) = sin(pos/10000^(2i/d)), PE(pos,2i+1) = cos(pos/10000^(2i/d)). Learned positions alternative.',
            
            'llm training': 'LLM training: 1) Pre-training on massive text corpora using self-supervised objectives (next token prediction, masked LM). 2) Supervised fine-tuning on instruction-following data. 3) RLHF for alignment with human preferences. Requires GPU clusters, months.',
            
            'rlhf': 'RLHF (Reinforcement Learning from Human Feedback) aligns LLMs with human preferences. Steps: 1) Supervised fine-tuning 2) Train reward model from human comparisons 3) Optimize policy with PPO using reward model. Developed by OpenAI, used in ChatGPT.',
            
            'lora': 'LoRA (Low-Rank Adaptation) efficiently fine-tunes LLMs by adding trainable low-rank matrices to frozen model weights. Reduces trainable parameters by 10,000x while maintaining performance. Updates: W\' = W + BA where B, A are low-rank.',
            
            'quantization': 'Quantization reduces model size and inference cost by using lower precision (int8, int4) instead of float32. Post-training quantization (PTQ), quantization-aware training (QAT). Techniques: GPTQ, AWQ, GGUF. 4-bit can reduce size 4-8x.',
            
            'inference optimization': 'LLM inference optimization: KV-cache (avoid recomputing attention), FlashAttention (memory-efficient attention), speculative decoding (draft model + verification), continuous batching, quantization, model parallelism, tensor parallelism.',
            
            'hallucination': 'Hallucination: LLMs generating plausible but incorrect/fabricated information. Causes: training data gaps, pattern completion, confidence calibration. Mitigation: RAG, fact-checking, calibration, instruction tuning, prompting for citations, uncertainty quantification.',
            
            'context window': 'Context window is maximum sequence length a model can process (measured in tokens). Limited by memory O(n²) and computation. GPT-4: 8k-128k tokens. Extending: sparse attention, memory mechanisms, retrieval. Longer contexts enable complex reasoning.',
            
            'temperature sampling': 'Temperature controls randomness in text generation. Low (0.1-0.5): focused, deterministic, consistent. High (0.8-1.5): creative, diverse, random. Temperature=0: greedy decoding (always most likely). Applied to logits before softmax.',
            
            'beam search': 'Beam search generates text by maintaining top-k most likely sequences at each step (beam width k). Balances quality and diversity better than greedy search. Alternatives: nucleus sampling (top-p), top-k sampling. Used in translation.',
            
            'model architecture': 'LLM architectures: Decoder-only (GPT - generation), Encoder-only (BERT - understanding), Encoder-decoder (T5 - seq2seq). Variations: Mixture of Experts (MoE - Mixtral), sparse models (Switch Transformer), multimodal (CLIP, Flamingo).',
            
            'scaling laws': 'Scaling laws: Model performance improves predictably with scale (parameters N, data D, compute C). Chinchilla optimal: balance model size and training tokens (~20 tokens per parameter). Emergent abilities appear at scale (e.g., reasoning).',
            
            'langchain': 'LangChain is a framework for building LLM applications. Provides: chains (sequential LLM calls), agents (autonomous task execution), memory (conversation history), tools (APIs, search), document loaders, vector store integration. Python & JavaScript.',
            
            'vector database': 'Vector databases store and query high-dimensional embeddings efficiently using ANN (approximate nearest neighbor) algorithms. Used in RAG for semantic search. Examples: Pinecone, Weaviate, Chroma, FAISS, Qdrant. Similarity: cosine, dot product, euclidean.',
            
            'semantic search': 'Semantic search finds similar documents using embedding similarity rather than keyword matching. Process: 1) Embed query 2) Compute similarity with document embeddings 3) Return top-k. Enables finding conceptually related content.',
            
            'llm agents': 'LLM agents autonomously execute tasks through planning, tool use, and iteration. Frameworks: ReAct (reasoning + acting), AutoGPT, BabyAGI. Components: LLM brain, tools (search, calculator, APIs), memory, reasoning loop. Challenges: reliability, costs.',
            
            'constitutional ai': 'Constitutional AI trains models to be helpful, harmless, honest using RL from AI Feedback (RLAIF) guided by principles (constitution). Self-critique and revision. Developed by Anthropic. Reduces reliance on human feedback.',
            
            'multimodal llm': 'Multimodal LLMs process multiple modalities: text, images, audio, video. Architecture: vision/audio encoder + language model. Examples: GPT-4V, Gemini, Claude 3. Applications: image captioning, VQA, document understanding.',
            
            'llm evaluation': 'LLM evaluation metrics: Perplexity (language modeling), BLEU/ROUGE/METEOR (generation), accuracy (classification), human evaluation. Benchmarks: MMLU (knowledge), HellaSwag (reasoning), HumanEval (coding), TruthfulQA (truthfulness), BigBench.',
            
            'prompt injection': 'Prompt injection: security vulnerability where malicious prompts override system instructions. Types: direct, indirect (through data). Defenses: input validation, output filtering, sandboxing, instruction hierarchy, Constitutional AI.',
            
            'llm distillation': 'Knowledge distillation trains smaller student models to mimic larger teacher models. Student learns from teacher\'s outputs (soft labels). Reduces size/cost while maintaining ~80-95% performance. Examples: DistilBERT, TinyBERT.',
            
            'chain of thought': 'Chain-of-thought (CoT) prompting encourages LLMs to show reasoning steps before answering. Improves performance on complex reasoning tasks. Variants: zero-shot CoT ("Let\'s think step by step"), few-shot CoT, self-consistency CoT.',
            
            'model compression': 'Model compression techniques: Pruning (remove weights), quantization (lower precision), distillation (smaller model), low-rank factorization (LoRA). Goal: reduce size, speed up inference, lower memory while maintaining accuracy.',
        }


# Example usage with the original interface
if __name__ == "__main__":
    # Initialize MockLLM
    llm = MockLLM()
    
    print("=== MockLLM - Comprehensive Knowledge Base ===\n")
    print("Compatible with BaseLLM interface\n")
    
    # Test queries
    test_queries = [
        "What is AI?",
        "Tell me about React hooks",
        "How does FastAPI work?",
        "Explain transformers in AI",
        "What is useEffect?",
        "How to handle authentication in FastAPI?",
        "What is RLHF?",
        "Explain React performance optimization",
        "What is RAG?",
        "How does dependency injection work?",
    ]
    
    for query in test_queries:
        print(f"Q: {query}")
        response = llm.generate(query)
        print(f"A: {response}\n")
        print("-" * 80 + "\n")