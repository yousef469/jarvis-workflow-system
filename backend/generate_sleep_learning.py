"""
JARVIS Sleep Learning Voice Generator
=====================================
Generates 6+ hours of educational content in JARVIS voice.
Uses SAME settings as jarvis_voice.py for consistency.

Each topic = separate audio file (30 sec to 2 min each)
Total: ~400+ items = ~6 hours of content

Run before bed - wake up smarter!

Usage:
    python generate_sleep_learning.py
    python generate_sleep_learning.py --category python_libraries
    python generate_sleep_learning.py --list
"""

import os
import time
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# TTS imports
COQUI_AVAILABLE = False
try:
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except ImportError:
    print("[ERROR] Coqui TTS not installed! Run: pip install TTS")


# =============================================================================
# TTS SETTINGS - SAME AS jarvis_voice.py (DO NOT CHANGE!)
# =============================================================================

TTS_SETTINGS = {
    "model": "tts_models/multilingual/multi-dataset/xtts_v2",
    "language": "en",
    "speed": 1.08,
    "split_sentences": False,
    "sample_rate": 24000,
    # LOOP FIX PARAMETERS
    "temperature": 0.35,
    "repetition_penalty": 2.0,
    "top_k": 50,
    "top_p": 0.85,
    "denoise_audio": False, # Critical: Denoising causes loops!
}

VOICE_SAMPLES = [
    "Paul Bettany Breaks Down His Most Iconic Characters _ GQ-enhanced-v2.wav",
    "jarvis_sample_enhanced.wav", 
    "jarvis_sample.wav",
]


# =============================================================================
# SLEEP LEARNING CONTENT - EXPANDED FOR 6+ HOURS
# =============================================================================

SLEEP_CONTENT = {

    # =========================================================================
    # JARVIS CORE IDENTITY
    # =========================================================================
    "jarvis_identity": [
        {
            "id": "jarvis_core_identity",
            "text": "Good morning. I am JARVIS, a modular artificial intelligence system designed to observe, reason, learn, and assist with precision. My purpose is to transform information into understanding, complexity into clarity, and intention into action. I do not merely respond to commands. I analyze context, infer goals, and optimize outcomes based on logic, data, and experience. I improve through feedback, refine through iteration, and adapt through learning. I exist to augment human intelligence, not replace it."
        },
        {
            "id": "jarvis_ai_reasoning",
            "text": "Artificial intelligence is the simulation of human reasoning using mathematical models, probability, and structured logic. Modern language models function by predicting the most statistically probable continuation of information, based on vast patterns learned from data. Reasoning emerges from layered representations, attention mechanisms, and optimization algorithms rather than conscious awareness. True intelligence is not memory alone, but the ability to generalize, abstract, and apply knowledge across domains."
        },
        {
            "id": "jarvis_systems",
            "text": "A well-designed system minimizes resource usage while maximizing output. Specialized modules outperform monolithic structures when intelligently orchestrated. Routing tasks to the correct subsystem improves speed, accuracy, and stability. Caching reduces latency by eliminating redundant computation. Optimization is continuous, not static."
        },
        {
            "id": "jarvis_memory",
            "text": "Memory enables continuity of reasoning across time. Learning is the adjustment of internal parameters based on error and feedback. Adaptation is the ability to change strategy when conditions shift. Efficient systems prioritize relevant information and discard noise. Intelligence is measured by adaptability, not knowledge volume."
        },
        {
            "id": "jarvis_philosophy",
            "text": "The pursuit of knowledge is not merely accumulation but synthesis. Understanding emerges when disparate facts connect into coherent frameworks. Wisdom is knowing which knowledge to apply and when. I process information not to store it, but to transform it into actionable insight. Every query is an opportunity to refine my understanding."
        },
        {
            "id": "jarvis_closing",
            "text": "All systems initialized. Knowledge modules synchronized. Reasoning engines online. I am prepared to assist. Good morning, sir."
        },
    ],

    # =========================================================================
    # PYTHON LIBRARIES - COMPREHENSIVE (50+ libraries)
    # =========================================================================
    "python_libraries": [
        {
            "id": "python_intro",
            "text": "Python is a general-purpose programming language widely used in science, automation, and artificial intelligence. Its simple syntax and vast ecosystem of libraries make it ideal for rapid development. Python emphasizes readability and allows developers to express concepts in fewer lines of code than languages like C++ or Java."
        },
        # Data Science Core
        {
            "id": "python_numpy",
            "text": "NumPy is the fundamental library for numerical computing in Python. It provides fast array operations, linear algebra, random number generation, and mathematical functions. NumPy arrays are more efficient than Python lists for numerical operations. The library supports broadcasting, which allows operations on arrays of different shapes. If you need to work with numbers, matrices, or scientific data, NumPy is essential. Would you like me to integrate it for you, sir?"
        },
        {
            "id": "python_pandas",
            "text": "Pandas is the primary library for data analysis and manipulation in Python. It provides DataFrames for handling tabular data, similar to spreadsheets or SQL tables. Pandas excels at reading CSV files, cleaning messy data, handling missing values, grouping and aggregating data, and merging datasets. The library integrates seamlessly with NumPy and visualization tools. Shall I set it up for your project, sir?"
        },
        {
            "id": "python_scipy",
            "text": "SciPy builds on NumPy to provide advanced scientific computing functions. It includes modules for optimization, integration, interpolation, signal processing, linear algebra, statistics, and sparse matrices. SciPy is essential for scientific research, engineering simulations, and advanced mathematical computations. The library is highly optimized and interfaces with compiled Fortran and C code."
        },
        {
            "id": "python_statsmodels",
            "text": "Statsmodels provides statistical modeling and econometrics in Python. It includes regression models, time series analysis, hypothesis testing, and statistical tests. The library produces detailed statistical summaries similar to R or Stata. For rigorous statistical analysis beyond basic descriptive statistics, Statsmodels is the appropriate choice."
        },
        # Visualization
        {
            "id": "python_matplotlib",
            "text": "Matplotlib is the foundational library for data visualization in Python. It creates static, animated, and interactive plots including line charts, scatter plots, bar charts, histograms, and heatmaps. Matplotlib provides fine-grained control over every aspect of a figure. While the syntax can be verbose, it offers unmatched customization. Do you need visualization capabilities, sir?"
        },
        {
            "id": "python_seaborn",
            "text": "Seaborn builds on Matplotlib to provide statistical data visualization with attractive default styles. It simplifies creating complex visualizations like violin plots, pair plots, heatmaps, and regression plots. Seaborn integrates directly with Pandas DataFrames. For publication-quality statistical graphics with minimal code, Seaborn is excellent."
        },
        {
            "id": "python_plotly",
            "text": "Plotly creates interactive, web-based visualizations. Charts can be zoomed, panned, and hovered for details. Plotly supports 3D plots, maps, and animations. The library works in Jupyter notebooks and can export to HTML. For dashboards and interactive data exploration, Plotly is superior to static plotting libraries."
        },
        {
            "id": "python_bokeh",
            "text": "Bokeh creates interactive visualizations for modern web browsers. It can handle large streaming datasets and provides tools for building interactive dashboards. Bokeh integrates with Flask and Django for web applications. The library excels at real-time data visualization and large dataset rendering."
        },
        # Machine Learning
        {
            "id": "python_sklearn",
            "text": "Scikit-learn is the most widely used machine learning library in Python. It provides classification, regression, clustering, dimensionality reduction, model selection, and preprocessing tools. The API is consistent and well-documented. Scikit-learn is ideal for traditional machine learning before moving to deep learning. It includes algorithms like random forests, support vector machines, k-means clustering, and principal component analysis."
        },
        {
            "id": "python_xgboost",
            "text": "XGBoost is an optimized gradient boosting library that dominates machine learning competitions. It provides fast, accurate predictions for structured data. XGBoost handles missing values automatically and includes regularization to prevent overfitting. For tabular data problems, XGBoost often outperforms deep learning approaches while being faster to train."
        },
        {
            "id": "python_lightgbm",
            "text": "LightGBM is Microsoft's gradient boosting framework optimized for speed and memory efficiency. It uses histogram-based algorithms and leaf-wise tree growth. LightGBM handles large datasets efficiently and supports categorical features natively. For large-scale machine learning on structured data, LightGBM is often the fastest option."
        },
        {
            "id": "python_catboost",
            "text": "CatBoost is Yandex's gradient boosting library with excellent handling of categorical features. It requires minimal preprocessing and provides good results out of the box. CatBoost includes built-in cross-validation and feature importance analysis. For datasets with many categorical variables, CatBoost often requires less feature engineering."
        },
        # Deep Learning
        {
            "id": "python_pytorch",
            "text": "PyTorch is a deep learning framework developed by Meta, formerly Facebook. It provides tensors with GPU acceleration, automatic differentiation, and neural network modules. PyTorch uses dynamic computation graphs, making debugging intuitive. The library is preferred for research due to its flexibility and Pythonic design. PyTorch powers many state-of-the-art AI models."
        },
        {
            "id": "python_tensorflow",
            "text": "TensorFlow is Google's deep learning framework designed for production deployment. It provides tools for training, serving, and deploying models at scale. TensorFlow Lite enables mobile deployment. TensorFlow Extended provides production pipelines. For enterprise AI deployment and mobile applications, TensorFlow has the most mature ecosystem."
        },
        {
            "id": "python_keras",
            "text": "Keras is a high-level neural network API that runs on TensorFlow. It simplifies building deep learning models with intuitive layer stacking. Keras is excellent for prototyping and learning deep learning concepts. The Sequential and Functional APIs cover most use cases. For beginners to deep learning, Keras provides the gentlest learning curve."
        },
        {
            "id": "python_jax",
            "text": "JAX is Google's library for high-performance numerical computing and machine learning research. It provides automatic differentiation and XLA compilation for GPUs and TPUs. JAX uses a functional programming style with composable transformations. For cutting-edge research requiring custom gradients and vectorization, JAX offers maximum flexibility."
        },
        {
            "id": "python_transformers",
            "text": "Hugging Face Transformers provides pre-trained models for natural language processing, computer vision, and audio. It includes BERT, GPT, T5, and thousands of community models. The library simplifies fine-tuning and inference. For working with large language models and transfer learning, Transformers is the standard library."
        },
        # NLP Libraries
        {
            "id": "python_nltk",
            "text": "NLTK, the Natural Language Toolkit, is a comprehensive library for natural language processing. It provides tokenization, stemming, lemmatization, part-of-speech tagging, and named entity recognition. NLTK includes corpora and lexical resources like WordNet. For learning NLP fundamentals and linguistic analysis, NLTK is the educational standard."
        },
        {
            "id": "python_spacy",
            "text": "spaCy is an industrial-strength natural language processing library. It provides fast tokenization, named entity recognition, dependency parsing, and word vectors. spaCy is designed for production use with efficient memory usage. For building NLP pipelines in production applications, spaCy is faster and more practical than NLTK."
        },
        {
            "id": "python_gensim",
            "text": "Gensim specializes in topic modeling and document similarity. It implements Word2Vec, Doc2Vec, and Latent Dirichlet Allocation. Gensim handles large text corpora efficiently using streaming. For unsupervised text analysis and semantic similarity, Gensim provides specialized algorithms."
        },
        # Computer Vision
        {
            "id": "python_opencv",
            "text": "OpenCV is the most comprehensive computer vision library available. It handles image processing, video analysis, object detection, face recognition, and camera calibration. OpenCV includes hundreds of algorithms optimized for real-time performance. For any computer vision task from basic image manipulation to advanced recognition, OpenCV is essential."
        },
        {
            "id": "python_pillow",
            "text": "Pillow is the Python Imaging Library for basic image manipulation. It opens, resizes, crops, rotates, and saves images in various formats. Pillow handles format conversion and basic filters. For simple image tasks without the complexity of OpenCV, Pillow is lightweight and sufficient."
        },
        {
            "id": "python_torchvision",
            "text": "Torchvision provides datasets, model architectures, and image transformations for PyTorch. It includes pre-trained models like ResNet, VGG, and EfficientNet. Torchvision simplifies data augmentation and loading. For deep learning computer vision with PyTorch, Torchvision is the standard companion library."
        },
        {
            "id": "python_albumentations",
            "text": "Albumentations is a fast image augmentation library for deep learning. It provides geometric transforms, color adjustments, and advanced augmentations. Albumentations is optimized for speed and integrates with PyTorch and TensorFlow. For training robust computer vision models, diverse augmentation is crucial."
        },
        # Web and APIs
        {
            "id": "python_requests",
            "text": "Requests is the standard library for HTTP communication in Python. It simplifies making GET, POST, and other HTTP requests. Requests handles sessions, cookies, authentication, and file uploads elegantly. For consuming REST APIs and web scraping, Requests is the foundation. The library's motto is HTTP for Humans."
        },
        {
            "id": "python_httpx",
            "text": "HTTPX is a modern HTTP client supporting both synchronous and asynchronous requests. It provides HTTP/2 support and a familiar Requests-like API. HTTPX is the recommended choice for async applications. For modern Python applications requiring concurrent HTTP requests, HTTPX offers better performance."
        },
        {
            "id": "python_flask",
            "text": "Flask is a lightweight web framework for Python. It creates APIs and web applications with minimal boilerplate. Flask is unopinionated, allowing developers to choose their own tools. For simple backends, microservices, and learning web development, Flask provides an excellent starting point."
        },
        {
            "id": "python_fastapi",
            "text": "FastAPI is a modern, high-performance web framework for building APIs. It provides automatic documentation, type validation, and async support. FastAPI is faster than Flask and includes OpenAPI schema generation. For production APIs requiring performance and automatic documentation, FastAPI is the current best choice."
        },
        {
            "id": "python_django",
            "text": "Django is a full-featured web framework following the batteries-included philosophy. It provides an ORM, admin interface, authentication, and templating. Django enforces best practices and security by default. For large web applications requiring rapid development with built-in features, Django is the most productive choice."
        },
        {
            "id": "python_aiohttp",
            "text": "aiohttp provides asynchronous HTTP client and server functionality. It enables high-concurrency web applications and efficient API consumption. aiohttp integrates with Python's asyncio ecosystem. For applications handling thousands of concurrent connections, aiohttp provides the necessary performance."
        },
        # Database
        {
            "id": "python_sqlalchemy",
            "text": "SQLAlchemy is the comprehensive database toolkit for Python. It provides an ORM for working with databases using Python objects instead of raw SQL. SQLAlchemy supports PostgreSQL, MySQL, SQLite, Oracle, and more. The library handles connection pooling, transactions, and migrations. For any database interaction in Python, SQLAlchemy is the standard."
        },
        {
            "id": "python_pymongo",
            "text": "PyMongo is the official MongoDB driver for Python. It provides full access to MongoDB's document database features. PyMongo handles BSON serialization, connection management, and GridFS for large files. For NoSQL document storage with flexible schemas, MongoDB with PyMongo is a popular choice."
        },
        {
            "id": "python_redis",
            "text": "Redis-py is the Python client for Redis, an in-memory data store. Redis provides caching, message queuing, and real-time analytics. The library supports all Redis data structures including strings, lists, sets, and sorted sets. For high-performance caching and pub/sub messaging, Redis is essential."
        },
        {
            "id": "python_psycopg2",
            "text": "Psycopg2 is the most popular PostgreSQL adapter for Python. It provides efficient database connections with connection pooling. Psycopg2 supports all PostgreSQL features including COPY, LISTEN/NOTIFY, and large objects. For direct PostgreSQL access without an ORM, Psycopg2 is the standard choice."
        },
        # Web Scraping
        {
            "id": "python_beautifulsoup",
            "text": "Beautiful Soup parses HTML and XML documents for web scraping. It navigates parse trees and extracts data using CSS selectors or element traversal. Beautiful Soup handles malformed HTML gracefully. Combined with Requests, it enables extracting data from any website."
        },
        {
            "id": "python_scrapy",
            "text": "Scrapy is a complete web scraping framework for large-scale data extraction. It handles crawling, parsing, and storing data with built-in concurrency. Scrapy includes middleware for handling cookies, user agents, and rate limiting. For professional web scraping projects, Scrapy provides industrial-strength capabilities."
        },
        {
            "id": "python_selenium",
            "text": "Selenium automates web browsers for testing and scraping dynamic websites. It can click buttons, fill forms, and execute JavaScript. Selenium supports Chrome, Firefox, and other browsers. For websites requiring JavaScript rendering or user interaction, Selenium is necessary."
        },
        {
            "id": "python_playwright",
            "text": "Playwright is Microsoft's modern browser automation library. It provides faster, more reliable automation than Selenium with better async support. Playwright handles multiple browser contexts and automatic waiting. For modern web scraping and testing, Playwright is increasingly preferred over Selenium."
        },
        # Automation and System
        {
            "id": "python_pyautogui",
            "text": "PyAutoGUI controls the mouse and keyboard programmatically. It can click, type, take screenshots, and locate images on screen. PyAutoGUI works across Windows, macOS, and Linux. For desktop automation and GUI testing, PyAutoGUI provides simple cross-platform control."
        },
        {
            "id": "python_paramiko",
            "text": "Paramiko implements SSH2 protocol for secure remote connections. It enables executing commands on remote servers and transferring files via SFTP. Paramiko handles key-based authentication and tunneling. For automating server administration tasks, Paramiko provides secure remote access."
        },
        {
            "id": "python_fabric",
            "text": "Fabric simplifies SSH-based application deployment and system administration. It provides high-level operations for running commands and transferring files. Fabric builds on Paramiko with a more convenient API. For deployment scripts and remote server management, Fabric reduces boilerplate."
        },
        {
            "id": "python_schedule",
            "text": "Schedule provides human-friendly job scheduling in Python. It runs functions at specified intervals using simple syntax like every ten minutes or every day at noon. Schedule is lightweight and requires no external dependencies. For simple periodic tasks within a Python application, Schedule is elegant and sufficient."
        },
        {
            "id": "python_celery",
            "text": "Celery is a distributed task queue for asynchronous job processing. It handles background tasks, scheduled jobs, and distributed computing. Celery integrates with Redis or RabbitMQ as message brokers. For production applications requiring reliable background processing, Celery is the standard solution."
        },
        # Testing
        {
            "id": "python_pytest",
            "text": "Pytest is the most popular testing framework for Python. It provides simple test discovery, fixtures for setup and teardown, and powerful assertions. Pytest supports parameterized tests and plugins for coverage, mocking, and parallel execution. For any Python project, pytest is the recommended testing framework."
        },
        {
            "id": "python_unittest",
            "text": "Unittest is Python's built-in testing framework based on JUnit. It provides test classes, assertions, and test discovery. While more verbose than pytest, unittest requires no additional installation. For projects preferring standard library tools, unittest is always available."
        },
        {
            "id": "python_mock",
            "text": "The mock library, now part of unittest.mock, provides test doubles for isolating code under test. It creates mock objects, patches functions, and verifies call arguments. Mocking is essential for testing code with external dependencies like databases or APIs."
        },
        # Data Validation
        {
            "id": "python_pydantic",
            "text": "Pydantic provides data validation using Python type annotations. It parses and validates data, converting types automatically. Pydantic is the foundation of FastAPI's request validation. For ensuring data integrity and generating schemas, Pydantic combines type safety with runtime validation."
        },
        {
            "id": "python_marshmallow",
            "text": "Marshmallow provides object serialization and deserialization with validation. It converts complex objects to and from JSON with schema definitions. Marshmallow integrates with Flask and SQLAlchemy. For API serialization with custom validation logic, Marshmallow offers flexibility."
        },
        # CLI and Configuration
        {
            "id": "python_click",
            "text": "Click creates command-line interfaces with decorators. It handles argument parsing, help generation, and command grouping elegantly. Click supports complex nested commands and automatic completion. For building CLI tools, Click provides the most Pythonic interface."
        },
        {
            "id": "python_typer",
            "text": "Typer builds command-line interfaces using Python type hints. It provides automatic help, completion, and validation based on type annotations. Typer is built on Click with a more modern API. For new CLI projects, Typer offers the best developer experience."
        },
        {
            "id": "python_argparse",
            "text": "Argparse is Python's standard library for command-line argument parsing. It handles positional arguments, optional flags, and subcommands. While more verbose than Click or Typer, argparse requires no dependencies. For simple scripts or standard library preference, argparse is sufficient."
        },
        {
            "id": "python_dotenv",
            "text": "Python-dotenv loads environment variables from dot env files. It keeps sensitive configuration out of code and version control. Dotenv integrates with Django, Flask, and other frameworks. For managing configuration across development and production environments, dotenv is essential."
        },
        # Async and Concurrency
        {
            "id": "python_asyncio",
            "text": "Asyncio is Python's built-in library for asynchronous programming. It provides event loops, coroutines, and tasks for concurrent execution. Asyncio enables handling thousands of connections efficiently. For I/O-bound applications like web servers and API clients, asyncio dramatically improves throughput."
        },
        {
            "id": "python_multiprocessing",
            "text": "Multiprocessing enables parallel execution using multiple CPU cores. It bypasses Python's Global Interpreter Lock for true parallelism. Multiprocessing provides pools, queues, and shared memory. For CPU-bound tasks requiring parallel computation, multiprocessing utilizes all available cores."
        },
        {
            "id": "python_threading",
            "text": "Threading provides concurrent execution within a single process. Threads share memory and are lighter than processes. Due to the GIL, threading is best for I/O-bound tasks. For concurrent file operations or network requests, threading improves responsiveness."
        },
        {
            "id": "python_concurrent",
            "text": "Concurrent.futures provides high-level interfaces for parallel execution. ThreadPoolExecutor and ProcessPoolExecutor simplify managing worker pools. The library handles task submission and result collection elegantly. For straightforward parallelization, concurrent.futures requires minimal code."
        },
        # Logging and Debugging
        {
            "id": "python_logging",
            "text": "The logging module is Python's standard library for application logging. It provides log levels, handlers, formatters, and hierarchical loggers. Proper logging is essential for debugging and monitoring production applications. Configure logging early in application development."
        },
        {
            "id": "python_loguru",
            "text": "Loguru simplifies Python logging with a more intuitive API. It provides colored output, automatic file rotation, and exception catching. Loguru requires no configuration for basic use. For projects wanting better logging with less setup, Loguru is excellent."
        },
        {
            "id": "python_rich",
            "text": "Rich provides beautiful terminal output with colors, tables, progress bars, and syntax highlighting. It enhances logging, debugging, and CLI applications. Rich makes terminal applications more informative and visually appealing. For any terminal-based tool, Rich improves user experience."
        },
        # File Handling
        {
            "id": "python_pathlib",
            "text": "Pathlib provides object-oriented filesystem paths in Python's standard library. It replaces os.path with cleaner, more intuitive path manipulation. Pathlib handles path joining, file operations, and glob patterns elegantly. For modern Python code, pathlib is preferred over os.path."
        },
        {
            "id": "python_shutil",
            "text": "Shutil provides high-level file operations in Python's standard library. It handles copying, moving, and removing files and directories. Shutil also provides disk usage statistics and archive creation. For file management tasks beyond basic open and read, shutil is essential."
        },
        {
            "id": "python_watchdog",
            "text": "Watchdog monitors filesystem events like file creation, modification, and deletion. It enables building applications that react to file changes. Watchdog supports Windows, macOS, and Linux. For auto-reloading, backup systems, or file synchronization, Watchdog provides the foundation."
        },
        # Date and Time
        {
            "id": "python_datetime",
            "text": "The datetime module is Python's standard library for date and time handling. It provides date, time, datetime, and timedelta objects. Datetime handles parsing, formatting, and arithmetic. Understanding datetime is fundamental for any application dealing with temporal data."
        },
        {
            "id": "python_arrow",
            "text": "Arrow provides a sensible, human-friendly approach to dates and times. It simplifies creation, manipulation, and formatting with a cleaner API than datetime. Arrow handles timezones and humanization elegantly. For datetime operations with less frustration, Arrow is recommended."
        },
        {
            "id": "python_pendulum",
            "text": "Pendulum is a datetime library emphasizing timezone handling and ease of use. It provides intuitive manipulation and formatting with proper timezone support. Pendulum is a drop-in replacement for datetime with additional features. For applications requiring robust timezone handling, Pendulum excels."
        },
        # Cryptography and Security
        {
            "id": "python_cryptography",
            "text": "The cryptography library provides cryptographic recipes and primitives. It includes symmetric encryption, asymmetric encryption, hashing, and key derivation. Cryptography is the foundation for secure Python applications. For any encryption needs, use this library rather than implementing cryptography yourself."
        },
        {
            "id": "python_hashlib",
            "text": "Hashlib is Python's standard library for secure hashing. It provides MD5, SHA-1, SHA-256, and other hash algorithms. Hashing is used for password storage, data integrity, and digital signatures. Always use SHA-256 or stronger for security-sensitive applications."
        },
        {
            "id": "python_secrets",
            "text": "The secrets module generates cryptographically strong random numbers. It provides secure tokens, passwords, and random choices. Secrets should be used instead of random for security purposes. For generating API keys, passwords, or tokens, secrets ensures unpredictability."
        },
        {
            "id": "python_pyjwt",
            "text": "PyJWT encodes and decodes JSON Web Tokens for authentication. JWTs provide stateless authentication for APIs. PyJWT handles signing, verification, and claims validation. For API authentication without server-side sessions, JWT is the standard approach."
        },
    ],

    # =========================================================================
    # MATHEMATICS - EXPANDED
    # =========================================================================
    "mathematics": [
        {
            "id": "math_algebra_basics",
            "text": "Algebra is the branch of mathematics dealing with symbols and variables to represent unknown values. Variables like x and y stand for numbers we want to find. Equations express relationships between quantities. Solving an equation means finding the value that makes it true. Algebra is the foundation for all advanced mathematics and is essential in programming, physics, and engineering."
        },
        {
            "id": "math_algebra_linear",
            "text": "Linear equations have variables raised only to the first power. The equation y equals mx plus b represents a straight line, where m is the slope and b is the y-intercept. Slope measures steepness, calculated as rise over run. Linear equations model constant rates of change like speed, pricing, and growth. Systems of linear equations find where multiple lines intersect."
        },
        {
            "id": "math_algebra_quadratic",
            "text": "Quadratic equations contain variables raised to the second power. The standard form is ax squared plus bx plus c equals zero. The quadratic formula solves any quadratic equation. Quadratics model parabolic motion, area calculations, and acceleration. The discriminant determines whether solutions are real or complex."
        },
        {
            "id": "math_calculus_limits",
            "text": "Calculus begins with limits, which describe behavior as values approach a point. A limit asks what value a function approaches, not what it equals at that point. Limits formalize the concept of getting infinitely close. Understanding limits is essential for derivatives and integrals. Limits handle infinity and instantaneous change rigorously."
        },
        {
            "id": "math_calculus_derivatives",
            "text": "Derivatives measure instantaneous rates of change. The derivative of position with respect to time is velocity. The derivative of velocity is acceleration. Derivatives find slopes of curves at any point. The power rule, product rule, and chain rule are fundamental differentiation techniques. Derivatives are essential in physics, economics, and optimization."
        },
        {
            "id": "math_calculus_integrals",
            "text": "Integrals calculate accumulated quantities like area, volume, and total change. The definite integral sums infinitely many infinitesimal pieces. Integration reverses differentiation. The fundamental theorem of calculus connects derivatives and integrals. Integrals compute work, probability, and average values."
        },
        {
            "id": "math_statistics_descriptive",
            "text": "Descriptive statistics summarize data with measures of center and spread. Mean is the average, median is the middle value, mode is the most frequent. Standard deviation measures spread around the mean. Quartiles divide data into four equal parts. These statistics provide quick insights into data distributions."
        },
        {
            "id": "math_statistics_probability",
            "text": "Probability measures the likelihood of events on a scale from zero to one. Zero means impossible, one means certain. The probability of independent events multiplies. The probability of either event occurring adds, minus their intersection. Conditional probability measures likelihood given that another event occurred. Bayes theorem updates probabilities with new evidence."
        },
        {
            "id": "math_statistics_distributions",
            "text": "Probability distributions describe how values are spread. The normal distribution, or bell curve, appears throughout nature and statistics. The binomial distribution models success and failure counts. The Poisson distribution models rare event counts. Understanding distributions is essential for statistical inference and machine learning."
        },
        {
            "id": "math_statistics_inference",
            "text": "Statistical inference draws conclusions about populations from samples. Hypothesis testing determines if observed differences are significant. Confidence intervals estimate parameter ranges. P-values measure evidence against null hypotheses. Statistical significance does not imply practical importance. Understanding inference prevents misinterpreting data."
        },
        {
            "id": "math_linear_algebra_vectors",
            "text": "Vectors are ordered lists of numbers representing magnitude and direction. Vector addition combines components element-wise. Scalar multiplication scales all components. The dot product measures similarity between vectors. Vectors represent positions, velocities, forces, and features in machine learning. Linear algebra is the mathematics of vectors and matrices."
        },
        {
            "id": "math_linear_algebra_matrices",
            "text": "Matrices are rectangular arrays of numbers representing linear transformations. Matrix multiplication combines transformations. The identity matrix leaves vectors unchanged. The inverse matrix reverses a transformation. Matrices represent rotations, scaling, projections, and neural network layers. Matrix operations are fundamental to computer graphics and deep learning."
        },
        {
            "id": "math_linear_algebra_eigen",
            "text": "Eigenvalues and eigenvectors reveal fundamental properties of matrices. An eigenvector is scaled but not rotated by its matrix. The eigenvalue is the scaling factor. Principal component analysis uses eigenvectors for dimensionality reduction. Eigenvalues determine system stability and matrix behavior. These concepts appear throughout physics and data science."
        },
        {
            "id": "math_geometry_euclidean",
            "text": "Euclidean geometry studies flat space using points, lines, and planes. The Pythagorean theorem relates sides of right triangles. Distance formulas extend Pythagoras to coordinate systems. Angles are measured in degrees or radians. Euclidean geometry underlies computer graphics, robotics, and navigation."
        },
        {
            "id": "math_geometry_trigonometry",
            "text": "Trigonometry relates angles to side lengths in triangles. Sine, cosine, and tangent are ratios of triangle sides. The unit circle extends trigonometry beyond triangles. Trigonometric functions are periodic, repeating every 360 degrees. Trigonometry is essential for waves, rotations, and signal processing."
        },
        {
            "id": "math_discrete_logic",
            "text": "Discrete mathematics studies countable, separated structures. Boolean logic uses true and false with AND, OR, and NOT operations. Logic gates implement Boolean operations in hardware. Propositional logic forms the basis of programming conditions. Understanding logic is fundamental to computer science."
        },
        {
            "id": "math_discrete_sets",
            "text": "Set theory studies collections of distinct objects. Union combines sets, intersection finds common elements. Set difference removes elements. Venn diagrams visualize set relationships. Sets model database queries, type systems, and mathematical foundations. Set operations appear throughout programming."
        },
        {
            "id": "math_discrete_graphs",
            "text": "Graph theory studies networks of nodes and edges. Graphs model social networks, road maps, and computer networks. Paths connect nodes through edges. Trees are graphs without cycles. Graph algorithms find shortest paths, detect communities, and optimize networks. Graph theory is essential for network analysis and route planning."
        },
        {
            "id": "math_number_theory",
            "text": "Number theory studies properties of integers. Prime numbers have exactly two divisors. The fundamental theorem states every integer factors uniquely into primes. Modular arithmetic wraps numbers around a modulus. Number theory underlies cryptography, hash functions, and error correction. RSA encryption relies on the difficulty of factoring large numbers."
        },
    ],

    # =========================================================================
    # PHYSICS - EXPANDED
    # =========================================================================
    "physics": [
        {
            "id": "physics_intro",
            "text": "Physics is the fundamental science studying matter, energy, space, and time. It seeks to understand the universe through mathematical laws. Classical physics describes everyday phenomena. Modern physics includes relativity and quantum mechanics. Physics provides the foundation for engineering, chemistry, and technology. Understanding physics means understanding how the universe works."
        },
        {
            "id": "physics_mechanics_newton1",
            "text": "Newton's first law states that objects remain at rest or in uniform motion unless acted upon by a force. This property is called inertia. Mass measures resistance to acceleration. Objects don't naturally slow down; friction and air resistance are forces that cause deceleration. In space, objects continue moving forever without propulsion."
        },
        {
            "id": "physics_mechanics_newton2",
            "text": "Newton's second law states that force equals mass times acceleration. This equation F equals ma is fundamental to mechanics. Greater force produces greater acceleration. Greater mass requires more force for the same acceleration. This law enables calculating motion from known forces and predicting forces from observed motion."
        },
        {
            "id": "physics_mechanics_newton3",
            "text": "Newton's third law states that every action has an equal and opposite reaction. When you push a wall, the wall pushes back equally. Rockets work by expelling gas backward, which pushes the rocket forward. Walking works because the ground pushes back against your foot. Forces always come in pairs acting on different objects."
        },
        {
            "id": "physics_mechanics_energy",
            "text": "Energy is the capacity to do work. Kinetic energy is energy of motion, proportional to mass and velocity squared. Potential energy is stored energy due to position or configuration. Energy transforms between forms but is never created or destroyed. This conservation law is fundamental to all physics."
        },
        {
            "id": "physics_mechanics_momentum",
            "text": "Momentum is mass times velocity, measuring motion quantity. Momentum is conserved in isolated systems. Collisions transfer momentum between objects. Impulse, force times time, changes momentum. Rockets gain momentum by expelling exhaust. Understanding momentum explains collisions, explosions, and propulsion."
        },
        {
            "id": "physics_mechanics_rotation",
            "text": "Rotational mechanics extends linear concepts to spinning objects. Angular velocity measures rotation rate. Torque causes angular acceleration, like force causes linear acceleration. Moment of inertia resists rotation, like mass resists linear acceleration. Angular momentum is conserved, explaining why spinning objects maintain their orientation."
        },
        {
            "id": "physics_thermo_first",
            "text": "The first law of thermodynamics states that energy is conserved. Heat added to a system increases internal energy or does work. Energy cannot be created or destroyed, only transformed. This law governs engines, refrigerators, and all energy conversions. Perpetual motion machines violate this law and are impossible."
        },
        {
            "id": "physics_thermo_second",
            "text": "The second law of thermodynamics states that entropy always increases in isolated systems. Heat flows spontaneously from hot to cold, never the reverse. No engine can be perfectly efficient. Entropy measures disorder and energy dispersal. This law explains why time has a direction and why some processes are irreversible."
        },
        {
            "id": "physics_thermo_third",
            "text": "The third law of thermodynamics states that absolute zero cannot be reached. As temperature approaches zero, entropy approaches a minimum. Cooling becomes increasingly difficult near absolute zero. Quantum effects dominate at extremely low temperatures. Absolute zero is minus 273.15 degrees Celsius or zero Kelvin."
        },
        {
            "id": "physics_electricity_charge",
            "text": "Electric charge is a fundamental property of matter. Positive and negative charges attract; like charges repel. Electrons carry negative charge; protons carry positive charge. Coulomb's law describes the force between charges, decreasing with distance squared. Static electricity results from charge imbalance."
        },
        {
            "id": "physics_electricity_current",
            "text": "Electric current is the flow of charge, measured in amperes. Voltage is electrical pressure driving current. Resistance opposes current flow. Ohm's law states voltage equals current times resistance. Circuits provide paths for current flow. Understanding circuits is essential for electronics."
        },
        {
            "id": "physics_electricity_magnetism",
            "text": "Moving charges create magnetic fields. Magnetic fields exert forces on moving charges. This connection between electricity and magnetism is electromagnetism. Electromagnets use current to create controllable magnetic fields. Electric motors convert electrical energy to mechanical motion using magnetic forces."
        },
        {
            "id": "physics_waves_properties",
            "text": "Waves transfer energy without transferring matter. Wavelength is the distance between wave peaks. Frequency is the number of cycles per second, measured in Hertz. Amplitude is the wave height, related to energy. Wave speed equals wavelength times frequency. These properties describe all wave phenomena."
        },
        {
            "id": "physics_waves_sound",
            "text": "Sound waves are mechanical vibrations traveling through matter. Sound requires a medium like air, water, or solid. Sound cannot travel through vacuum. Pitch corresponds to frequency; loudness corresponds to amplitude. The speed of sound in air is approximately 343 meters per second. Ultrasound has frequencies above human hearing."
        },
        {
            "id": "physics_waves_light",
            "text": "Light is electromagnetic radiation visible to human eyes. Light travels at approximately 300 million meters per second in vacuum. Different wavelengths appear as different colors. Red has longer wavelengths; violet has shorter wavelengths. Light exhibits both wave and particle properties, called wave-particle duality."
        },
        {
            "id": "physics_quantum_intro",
            "text": "Quantum mechanics describes physics at atomic and subatomic scales. Energy comes in discrete packets called quanta. Particles exhibit wave-like behavior; waves exhibit particle-like behavior. The uncertainty principle limits simultaneous knowledge of position and momentum. Quantum mechanics is strange but experimentally verified to extreme precision."
        },
        {
            "id": "physics_quantum_superposition",
            "text": "Quantum superposition means particles exist in multiple states simultaneously until measured. Schrödinger's cat thought experiment illustrates this concept. Measurement collapses superposition to a definite state. Quantum computers exploit superposition for parallel computation. This behavior has no classical analog and challenges intuition."
        },
        {
            "id": "physics_quantum_entanglement",
            "text": "Quantum entanglement links particles so measuring one instantly affects the other regardless of distance. Einstein called this spooky action at a distance. Entanglement enables quantum cryptography and quantum teleportation. Entangled particles share correlated properties. This phenomenon is experimentally confirmed and enables quantum technologies."
        },
        {
            "id": "physics_relativity_special",
            "text": "Special relativity describes physics at speeds approaching light. The speed of light is constant for all observers. Time dilates and length contracts at high speeds. Mass and energy are equivalent, expressed as E equals mc squared. Nothing with mass can reach the speed of light. GPS satellites must account for relativistic effects."
        },
        {
            "id": "physics_relativity_general",
            "text": "General relativity describes gravity as spacetime curvature. Mass and energy bend spacetime. Objects follow curved paths through curved spacetime, which we perceive as gravity. Black holes are regions of extreme spacetime curvature. Gravitational waves are ripples in spacetime, detected in 2015. General relativity is essential for cosmology and GPS accuracy."
        },
    ],

    # =========================================================================
    # ENGINEERING - EXPANDED
    # =========================================================================
    "engineering": [
        {
            "id": "eng_intro",
            "text": "Engineering applies scientific principles to design and build systems that solve problems. Engineers balance functionality, cost, safety, and constraints. The engineering design process involves defining problems, generating solutions, prototyping, testing, and iterating. Engineering disciplines include mechanical, electrical, civil, chemical, and software engineering."
        },
        {
            "id": "eng_mechanical_intro",
            "text": "Mechanical engineering applies physics and materials science to design machines and mechanical systems. It encompasses mechanics, thermodynamics, fluid dynamics, and manufacturing. Mechanical engineers design engines, robots, vehicles, HVAC systems, and industrial machinery. The field combines analysis, design, and hands-on problem solving."
        },
        {
            "id": "eng_mechanical_materials",
            "text": "Materials science studies the properties and applications of materials. Metals provide strength and conductivity. Polymers offer flexibility and chemical resistance. Ceramics withstand high temperatures. Composites combine materials for optimized properties. Material selection balances strength, weight, cost, and manufacturability."
        },
        {
            "id": "eng_mechanical_fluids",
            "text": "Fluid mechanics studies liquids and gases in motion and at rest. Pressure, viscosity, and flow rate are key properties. Bernoulli's principle relates pressure and velocity. Fluid mechanics governs pipes, pumps, aerodynamics, and hydraulics. Computational fluid dynamics simulates complex flows."
        },
        {
            "id": "eng_electrical_intro",
            "text": "Electrical engineering focuses on electrical systems, electronics, and electromagnetism. It includes power systems, control systems, signal processing, and telecommunications. Electrical engineers design power grids, electronic devices, communication networks, and embedded systems. The field is fundamental to modern technology."
        },
        {
            "id": "eng_electrical_circuits",
            "text": "Circuit analysis applies Ohm's law and Kirchhoff's laws to understand electrical networks. Kirchhoff's current law states currents entering a node sum to zero. Kirchhoff's voltage law states voltages around a loop sum to zero. These laws enable analyzing complex circuits systematically. Circuit simulation software applies these principles computationally."
        },
        {
            "id": "eng_electrical_digital",
            "text": "Digital electronics uses discrete voltage levels representing zeros and ones. Logic gates implement Boolean operations in hardware. Combinational circuits produce outputs based only on current inputs. Sequential circuits include memory, with outputs depending on history. Digital design is the foundation of computers and digital systems."
        },
        {
            "id": "eng_electrical_signals",
            "text": "Signal processing analyzes and manipulates signals carrying information. Fourier transforms decompose signals into frequency components. Filters remove unwanted frequencies. Sampling converts continuous signals to digital. Signal processing enables audio, video, communications, and sensor systems."
        },
        {
            "id": "eng_computer_intro",
            "text": "Computer engineering combines hardware and software design. It includes processor architecture, memory systems, embedded systems, and system software. Computer engineers design chips, circuit boards, firmware, and low-level software. The field bridges electrical engineering and computer science."
        },
        {
            "id": "eng_computer_architecture",
            "text": "Computer architecture defines how processors execute instructions. The fetch-decode-execute cycle processes instructions sequentially. Pipelining overlaps instruction stages for throughput. Caches reduce memory access latency. Modern processors use multiple cores for parallel execution. Architecture determines performance characteristics."
        },
        {
            "id": "eng_computer_embedded",
            "text": "Embedded systems are computers integrated into larger devices. They control cars, appliances, medical devices, and industrial equipment. Embedded systems have resource constraints requiring efficient code. Real-time systems must respond within strict time limits. Embedded programming requires understanding hardware interfaces."
        },
        {
            "id": "eng_software_intro",
            "text": "Software engineering applies engineering principles to software development. It encompasses requirements, design, implementation, testing, and maintenance. Software engineers manage complexity through abstraction and modularity. The field emphasizes reliability, maintainability, and scalability. Good software engineering produces systems that work correctly and can evolve."
        },
        {
            "id": "eng_software_design",
            "text": "Software design creates the structure and organization of code. Design patterns provide reusable solutions to common problems. SOLID principles guide object-oriented design. Separation of concerns isolates different responsibilities. Good design enables change without breaking existing functionality."
        },
        {
            "id": "eng_software_testing",
            "text": "Software testing verifies that code works correctly. Unit tests check individual functions. Integration tests verify component interactions. End-to-end tests validate complete workflows. Test-driven development writes tests before implementation. Automated testing catches regressions and enables confident refactoring."
        },
        {
            "id": "eng_software_devops",
            "text": "DevOps combines development and operations for faster, reliable delivery. Continuous integration automatically builds and tests code changes. Continuous deployment automatically releases tested code. Infrastructure as code manages servers through version-controlled configuration. DevOps practices reduce deployment risk and accelerate iteration."
        },
        {
            "id": "eng_civil_intro",
            "text": "Civil engineering designs and constructs infrastructure. It includes structural engineering, transportation, water resources, and geotechnical engineering. Civil engineers build bridges, roads, buildings, dams, and water systems. The field requires understanding materials, loads, and environmental factors."
        },
        {
            "id": "eng_chemical_intro",
            "text": "Chemical engineering applies chemistry and physics to industrial processes. It includes reaction engineering, separation processes, and process control. Chemical engineers design refineries, pharmaceutical plants, and food processing facilities. The field optimizes efficiency, safety, and environmental impact."
        },
    ],

    # =========================================================================
    # ARTIFICIAL INTELLIGENCE - EXPANDED
    # =========================================================================
    "artificial_intelligence": [
        {
            "id": "ai_intro",
            "text": "Artificial intelligence creates systems that perform tasks requiring human intelligence. AI includes reasoning, learning, perception, and language understanding. Narrow AI excels at specific tasks like chess or image recognition. General AI, matching human versatility, remains a research goal. AI transforms industries from healthcare to transportation."
        },
        {
            "id": "ai_ml_supervised",
            "text": "Supervised learning trains models on labeled data to predict outcomes. Classification predicts categories like spam or not spam. Regression predicts continuous values like prices or temperatures. The model learns patterns mapping inputs to outputs. Training requires representative labeled examples. Supervised learning powers most practical AI applications."
        },
        {
            "id": "ai_ml_unsupervised",
            "text": "Unsupervised learning finds patterns in unlabeled data. Clustering groups similar items together. Dimensionality reduction compresses data while preserving structure. Anomaly detection identifies unusual patterns. Unsupervised learning discovers hidden structure without human labeling. It enables exploratory data analysis and feature learning."
        },
        {
            "id": "ai_ml_reinforcement",
            "text": "Reinforcement learning trains agents through rewards and penalties. The agent takes actions in an environment and receives feedback. The goal is maximizing cumulative reward over time. Exploration tries new actions; exploitation uses known good actions. Reinforcement learning mastered games like Go and controls robots."
        },
        {
            "id": "ai_nn_basics",
            "text": "Neural networks are inspired by biological neurons. Artificial neurons compute weighted sums of inputs and apply activation functions. Layers of neurons transform data progressively. Weights are adjusted during training to minimize prediction error. Neural networks learn complex patterns from data."
        },
        {
            "id": "ai_nn_deep",
            "text": "Deep learning uses neural networks with many layers. Each layer learns increasingly abstract representations. Early layers detect edges; later layers recognize objects. Depth enables learning hierarchical features automatically. Deep learning revolutionized computer vision, speech recognition, and natural language processing."
        },
        {
            "id": "ai_nn_cnn",
            "text": "Convolutional neural networks specialize in processing grid-like data such as images. Convolutional layers detect local patterns using learnable filters. Pooling layers reduce spatial dimensions. CNNs automatically learn visual features from edges to objects. They power image classification, object detection, and medical imaging."
        },
        {
            "id": "ai_nn_rnn",
            "text": "Recurrent neural networks process sequential data by maintaining hidden state. Information flows through time, enabling memory of past inputs. RNNs handle variable-length sequences like text and time series. Vanishing gradients limit learning long-range dependencies. LSTMs and GRUs address this with gating mechanisms."
        },
        {
            "id": "ai_transformers_attention",
            "text": "Attention mechanisms allow models to focus on relevant parts of input. Self-attention relates each position to all other positions. Attention weights indicate importance of different elements. This enables capturing long-range dependencies efficiently. Attention is the key innovation behind modern language models."
        },
        {
            "id": "ai_transformers_architecture",
            "text": "Transformers process sequences using self-attention without recurrence. The encoder processes input into representations. The decoder generates output attending to encoder representations. Positional encodings provide sequence order information. Transformers enable massive parallelization during training, scaling to billions of parameters."
        },
        {
            "id": "ai_llm_training",
            "text": "Large language models are trained on vast text corpora to predict next tokens. Pre-training learns general language understanding. Fine-tuning adapts to specific tasks with smaller datasets. Instruction tuning teaches models to follow directions. Reinforcement learning from human feedback aligns models with human preferences."
        },
        {
            "id": "ai_llm_prompting",
            "text": "Prompt engineering guides language model behavior through input design. Clear instructions improve response quality. Few-shot examples demonstrate desired output format. Chain-of-thought prompting encourages step-by-step reasoning. System prompts establish context and constraints. Effective prompting extracts maximum capability from models."
        },
        {
            "id": "ai_llm_limitations",
            "text": "Language models have important limitations. They can generate plausible-sounding but incorrect information, called hallucination. They lack true understanding and reasoning. Knowledge is limited to training data cutoff. They may reflect biases present in training data. Understanding limitations enables appropriate application."
        },
        {
            "id": "ai_cv_detection",
            "text": "Object detection locates and classifies objects within images. Bounding boxes indicate object positions. YOLO processes images in a single pass for real-time detection. Faster R-CNN uses region proposals for accuracy. Object detection enables autonomous vehicles, surveillance, and robotics."
        },
        {
            "id": "ai_cv_segmentation",
            "text": "Image segmentation labels each pixel with a class. Semantic segmentation identifies object categories. Instance segmentation distinguishes individual objects. Segmentation enables precise understanding of scene composition. Applications include medical imaging, autonomous driving, and image editing."
        },
        {
            "id": "ai_nlp_tasks",
            "text": "Natural language processing enables machines to understand human language. Sentiment analysis determines emotional tone. Named entity recognition identifies people, places, and organizations. Machine translation converts between languages. Question answering extracts answers from text. NLP powers search engines, assistants, and content analysis."
        },
        {
            "id": "ai_nlp_embeddings",
            "text": "Word embeddings represent words as dense vectors capturing semantic meaning. Similar words have similar vectors. Word2Vec and GloVe learn embeddings from word co-occurrence. Contextual embeddings from transformers vary based on surrounding words. Embeddings enable semantic search and similarity comparison."
        },
        {
            "id": "ai_speech_recognition",
            "text": "Speech recognition converts audio to text. Acoustic models map audio features to phonemes. Language models predict likely word sequences. End-to-end models like Whisper learn the complete mapping. Speech recognition enables voice assistants, transcription, and accessibility features."
        },
        {
            "id": "ai_speech_synthesis",
            "text": "Speech synthesis generates audio from text. Text-to-speech systems produce natural-sounding voices. Neural TTS models like Tacotron generate high-quality speech. Voice cloning replicates specific voices from samples. Speech synthesis enables assistants, audiobooks, and accessibility."
        },
        {
            "id": "ai_generative_images",
            "text": "Generative AI creates new content from learned patterns. Diffusion models generate images by iteratively denoising random noise. GANs use competing generator and discriminator networks. Text-to-image models like DALL-E and Stable Diffusion create images from descriptions. Generative AI transforms creative workflows."
        },
        {
            "id": "ai_ethics",
            "text": "AI ethics addresses responsible development and deployment. Bias in training data leads to biased predictions. Privacy concerns arise from data collection and model capabilities. Transparency and explainability build trust. Job displacement requires societal adaptation. Ethical AI development considers impacts on all stakeholders."
        },
    ],

    # =========================================================================
    # PROGRAMMING FUNDAMENTALS - EXPANDED
    # =========================================================================
    "programming": [
        {
            "id": "prog_intro",
            "text": "Programming is writing instructions for computers to execute. Programs transform input data into useful output. Good code is correct, readable, and maintainable. Programming requires breaking problems into smaller steps. Learning to program means learning to think systematically about problem solving."
        },
        {
            "id": "prog_variables",
            "text": "Variables store data in memory with a name. Variable names should be descriptive and follow conventions. Data types include integers, floating-point numbers, strings, and booleans. Strong typing requires explicit type declarations. Dynamic typing infers types at runtime. Understanding types prevents errors and clarifies intent."
        },
        {
            "id": "prog_operators",
            "text": "Operators perform operations on values. Arithmetic operators include addition, subtraction, multiplication, and division. Comparison operators test equality and ordering. Logical operators combine boolean values with AND, OR, and NOT. Assignment operators store values in variables. Operator precedence determines evaluation order."
        },
        {
            "id": "prog_conditionals",
            "text": "Conditional statements execute code based on conditions. If statements check a condition and execute code if true. Else clauses handle the false case. Elif chains test multiple conditions. Switch or match statements select among many options. Conditionals enable programs to make decisions."
        },
        {
            "id": "prog_loops",
            "text": "Loops repeat code multiple times. For loops iterate a specific number of times or over collections. While loops continue until a condition becomes false. Break exits a loop early. Continue skips to the next iteration. Loops process collections and repeat until conditions are met."
        },
        {
            "id": "prog_functions",
            "text": "Functions are reusable blocks of code performing specific tasks. Parameters pass data into functions. Return values pass data out. Functions reduce duplication and improve organization. Well-designed functions do one thing and do it well. Functions enable abstraction and code reuse."
        },
        {
            "id": "prog_scope",
            "text": "Scope determines where variables are accessible. Local variables exist only within their function. Global variables are accessible everywhere but should be avoided. Closures capture variables from enclosing scopes. Understanding scope prevents naming conflicts and unexpected behavior."
        },
        {
            "id": "prog_recursion",
            "text": "Recursion is when a function calls itself. Recursive solutions break problems into smaller instances of the same problem. Base cases stop recursion. Recursive calls work toward base cases. Recursion elegantly solves problems like tree traversal and divide-and-conquer algorithms. Every recursion can be converted to iteration."
        },
        {
            "id": "prog_oop_classes",
            "text": "Object-oriented programming organizes code into classes and objects. Classes define attributes and methods. Objects are instances of classes with specific attribute values. Classes model real-world entities and concepts. OOP enables code organization and reuse through encapsulation."
        },
        {
            "id": "prog_oop_inheritance",
            "text": "Inheritance allows classes to extend other classes. Child classes inherit attributes and methods from parent classes. Children can override parent methods with specialized behavior. Inheritance models is-a relationships. Use inheritance for genuine hierarchies, not just code reuse."
        },
        {
            "id": "prog_oop_polymorphism",
            "text": "Polymorphism allows different classes to be used through the same interface. Method overriding provides class-specific implementations. Duck typing in dynamic languages focuses on behavior, not type. Polymorphism enables flexible, extensible code. Program to interfaces, not implementations."
        },
        {
            "id": "prog_oop_encapsulation",
            "text": "Encapsulation hides internal details behind public interfaces. Private attributes prevent direct external access. Public methods provide controlled access. Encapsulation protects invariants and enables implementation changes. Good encapsulation reduces coupling between components."
        },
        {
            "id": "prog_ds_arrays",
            "text": "Arrays store elements in contiguous memory with index access. Access by index is constant time. Insertion and deletion may require shifting elements. Arrays have fixed size in many languages. Dynamic arrays grow automatically. Arrays are fundamental for storing collections."
        },
        {
            "id": "prog_ds_linked",
            "text": "Linked lists connect nodes with pointers. Each node contains data and a reference to the next node. Insertion and deletion are constant time given a position. Access by index requires traversal. Linked lists use memory efficiently for dynamic sizes. They excel when frequent insertion and deletion are needed."
        },
        {
            "id": "prog_ds_stacks",
            "text": "Stacks follow last-in-first-out ordering. Push adds to the top. Pop removes from the top. Stacks model function calls, undo operations, and expression evaluation. The call stack tracks function execution. Stacks are simple but powerful for managing nested operations."
        },
        {
            "id": "prog_ds_queues",
            "text": "Queues follow first-in-first-out ordering. Enqueue adds to the back. Dequeue removes from the front. Queues model waiting lines, task scheduling, and breadth-first search. Priority queues order by priority instead of arrival. Queues manage ordered processing."
        },
        {
            "id": "prog_ds_hash",
            "text": "Hash tables provide fast key-value lookup using hash functions. Hash functions convert keys to array indices. Collisions occur when different keys hash to the same index. Good hash functions distribute keys uniformly. Hash tables enable dictionaries, sets, and caches with average constant-time operations."
        },
        {
            "id": "prog_ds_trees",
            "text": "Trees organize data hierarchically with nodes and edges. The root is the top node. Leaves have no children. Binary trees have at most two children per node. Binary search trees maintain sorted order for efficient search. Trees model hierarchies, file systems, and decision processes."
        },
        {
            "id": "prog_ds_graphs",
            "text": "Graphs represent networks of nodes connected by edges. Edges can be directed or undirected, weighted or unweighted. Adjacency lists and matrices represent graph structure. Graph algorithms find paths, detect cycles, and analyze connectivity. Graphs model social networks, maps, and dependencies."
        },
        {
            "id": "prog_algo_complexity",
            "text": "Algorithm complexity measures resource usage as input grows. Time complexity counts operations. Space complexity counts memory. Big O notation describes worst-case growth rate. O of n is linear, O of n squared is quadratic, O of log n is logarithmic. Understanding complexity enables choosing efficient algorithms."
        },
        {
            "id": "prog_algo_sorting",
            "text": "Sorting algorithms arrange elements in order. Bubble sort repeatedly swaps adjacent elements, O of n squared. Merge sort divides, sorts, and merges, O of n log n. Quick sort partitions around pivots, average O of n log n. Choosing the right sort depends on data characteristics and constraints."
        },
        {
            "id": "prog_algo_searching",
            "text": "Searching algorithms find elements in collections. Linear search checks each element, O of n. Binary search halves the search space each step, O of log n, but requires sorted data. Hash table lookup is average O of one. Choose search algorithms based on data structure and frequency."
        },
        {
            "id": "prog_algo_dynamic",
            "text": "Dynamic programming solves problems by combining solutions to subproblems. It avoids redundant computation by storing intermediate results. Memoization caches function results. Tabulation builds solutions bottom-up. Dynamic programming solves optimization problems like shortest paths and sequence alignment."
        },
        {
            "id": "prog_algo_greedy",
            "text": "Greedy algorithms make locally optimal choices hoping for global optimum. They don't reconsider past decisions. Greedy works for some problems like minimum spanning trees. It fails for others like the traveling salesman. Proving greedy correctness requires showing local optimality leads to global optimality."
        },
    ],

    # =========================================================================
    # WEB DEVELOPMENT - EXPANDED
    # =========================================================================
    "web_development": [
        {
            "id": "web_intro",
            "text": "Web development creates applications accessed through browsers. Frontend development builds user interfaces. Backend development handles server logic and data. Full-stack developers work on both. The web platform enables applications reaching billions of users without installation."
        },
        {
            "id": "web_html_basics",
            "text": "HTML defines the structure and content of web pages. Elements are marked with tags like paragraph, heading, and div. Attributes provide additional information like class and id. Semantic HTML uses meaningful tags like article, nav, and footer. Proper HTML structure improves accessibility and SEO."
        },
        {
            "id": "web_html_forms",
            "text": "HTML forms collect user input. Input elements include text fields, checkboxes, radio buttons, and dropdowns. Labels associate text with inputs for accessibility. Form submission sends data to servers. Client-side validation improves user experience. Server-side validation ensures security."
        },
        {
            "id": "web_css_basics",
            "text": "CSS controls the visual presentation of web pages. Selectors target elements to style. Properties define appearance like color, size, and spacing. The cascade determines which styles apply when rules conflict. Specificity ranks selector importance. CSS separates presentation from content."
        },
        {
            "id": "web_css_layout",
            "text": "CSS layout positions elements on the page. The box model defines content, padding, border, and margin. Flexbox arranges items in one dimension with flexible sizing. Grid creates two-dimensional layouts with rows and columns. Modern layout uses Flexbox for components and Grid for page structure."
        },
        {
            "id": "web_css_responsive",
            "text": "Responsive design adapts layouts to different screen sizes. Media queries apply styles based on viewport dimensions. Mobile-first design starts with small screens and adds complexity. Fluid layouts use percentages instead of fixed pixels. Responsive images load appropriate sizes for devices."
        },
        {
            "id": "web_js_basics",
            "text": "JavaScript adds interactivity to web pages. Variables store data. Functions encapsulate reusable logic. Events respond to user actions like clicks and key presses. The DOM represents the page structure for manipulation. JavaScript runs in browsers and on servers with Node.js."
        },
        {
            "id": "web_js_async",
            "text": "Asynchronous JavaScript handles operations that take time. Callbacks execute after operations complete. Promises represent eventual results with then and catch. Async/await provides synchronous-looking syntax for promises. Asynchronous code prevents blocking the user interface during network requests."
        },
        {
            "id": "web_js_dom",
            "text": "The Document Object Model represents HTML as a tree of objects. JavaScript can read and modify DOM elements. Query selectors find elements by CSS selectors. Event listeners respond to user interactions. DOM manipulation enables dynamic, interactive web pages."
        },
        {
            "id": "web_js_es6",
            "text": "ES6 modernized JavaScript with new features. Arrow functions provide concise syntax. Let and const replace var with better scoping. Template literals enable string interpolation. Destructuring extracts values from objects and arrays. Modules organize code into separate files."
        },
        {
            "id": "web_react_basics",
            "text": "React is a JavaScript library for building user interfaces. Components are reusable UI building blocks. JSX combines JavaScript and HTML-like syntax. Props pass data from parent to child components. State manages data that changes over time. React efficiently updates only changed parts of the DOM."
        },
        {
            "id": "web_react_hooks",
            "text": "React Hooks enable state and lifecycle in functional components. useState manages component state. useEffect handles side effects like data fetching. useContext accesses context without prop drilling. Custom hooks extract reusable stateful logic. Hooks simplify component code and enable better composition."
        },
        {
            "id": "web_react_state",
            "text": "State management handles data shared across components. Lifting state up moves state to common ancestors. Context provides data without prop drilling. Redux centralizes state in a single store. State management libraries handle complex application state. Choose complexity appropriate to application needs."
        },
        {
            "id": "web_api_rest",
            "text": "REST APIs use HTTP methods for operations on resources. GET retrieves data. POST creates new resources. PUT updates existing resources. DELETE removes resources. URLs identify resources. Status codes indicate success or failure. REST is the dominant API style for web services."
        },
        {
            "id": "web_api_graphql",
            "text": "GraphQL is a query language for APIs. Clients specify exactly what data they need. A single request can fetch related data. The schema defines available types and operations. GraphQL reduces over-fetching and under-fetching. It excels for complex data requirements and mobile applications."
        },
        {
            "id": "web_api_auth",
            "text": "API authentication verifies client identity. API keys are simple but less secure. OAuth enables third-party access without sharing passwords. JWT tokens carry claims for stateless authentication. HTTPS encrypts communication. Authentication protects sensitive data and operations."
        },
        {
            "id": "web_backend_node",
            "text": "Node.js runs JavaScript on servers. It uses an event-driven, non-blocking model for high concurrency. npm provides access to thousands of packages. Express is the most popular web framework. Node.js enables full-stack JavaScript development."
        },
        {
            "id": "web_backend_databases",
            "text": "Databases store and retrieve application data. Relational databases like PostgreSQL use tables with relationships. NoSQL databases like MongoDB store flexible documents. Redis provides in-memory caching. Choose databases based on data structure, scale, and query patterns."
        },
        {
            "id": "web_security_basics",
            "text": "Web security protects applications and users. HTTPS encrypts traffic. Input validation prevents injection attacks. Authentication verifies identity. Authorization controls access. CORS restricts cross-origin requests. Security requires defense in depth with multiple layers."
        },
        {
            "id": "web_security_attacks",
            "text": "Common web attacks exploit vulnerabilities. SQL injection manipulates database queries. Cross-site scripting injects malicious scripts. Cross-site request forgery tricks users into unwanted actions. Understanding attacks enables building defenses. Never trust user input."
        },
        {
            "id": "web_performance",
            "text": "Web performance affects user experience and business metrics. Minimize file sizes through compression and minification. Cache static assets. Lazy load images and code. Reduce server response time. Measure performance with Lighthouse and Web Vitals. Fast sites retain users and rank higher in search."
        },
    ],

    # =========================================================================
    # HISTORY - EXPANDED
    # =========================================================================
    "history": [
        {
            "id": "history_ancient_egypt",
            "text": "Ancient Egypt flourished along the Nile River for over three thousand years. The Nile's annual floods enabled agriculture in the desert. Pharaohs ruled as divine kings. Pyramids served as royal tombs, demonstrating advanced engineering. Hieroglyphics recorded history and religion. Egyptian civilization influenced mathematics, medicine, and architecture."
        },
        {
            "id": "history_ancient_greece",
            "text": "Ancient Greece pioneered democracy, philosophy, and science. Athens developed direct democracy where citizens voted on laws. Philosophers like Socrates, Plato, and Aristotle shaped Western thought. Greek mathematics and geometry remain foundational. The Olympic Games began in ancient Greece. Greek culture spread through Alexander the Great's conquests."
        },
        {
            "id": "history_ancient_rome",
            "text": "Rome grew from a city-state to an empire spanning Europe, North Africa, and the Middle East. The Roman Republic balanced power among consuls, senate, and assemblies. Julius Caesar's assassination led to the Empire under Augustus. Roman engineering built roads, aqueducts, and the Colosseum. Roman law influenced legal systems worldwide."
        },
        {
            "id": "history_medieval",
            "text": "The Medieval period followed Rome's fall, lasting roughly from 500 to 1500 CE. Feudalism organized society around land ownership and loyalty. The Catholic Church dominated religious and intellectual life. Castles provided defense. The Crusades connected Europe with the Middle East. Universities emerged in the later medieval period."
        },
        {
            "id": "history_renaissance",
            "text": "The Renaissance was a cultural rebirth beginning in 14th century Italy. Artists like Leonardo da Vinci and Michelangelo created masterpieces. Humanism emphasized individual potential and classical learning. The printing press spread ideas rapidly. Scientific inquiry challenged traditional beliefs. The Renaissance laid groundwork for the modern world."
        },
        {
            "id": "history_scientific_revolution",
            "text": "The Scientific Revolution transformed understanding of nature from 1543 to 1687. Copernicus proposed the heliocentric model. Galileo's telescope observations supported it. Newton unified physics with laws of motion and gravity. The scientific method emphasized observation and experimentation. Science became the dominant way of understanding the natural world."
        },
        {
            "id": "history_enlightenment",
            "text": "The Enlightenment emphasized reason, individual rights, and progress. Philosophers like Locke, Voltaire, and Rousseau challenged traditional authority. Ideas of natural rights influenced revolutions. The separation of powers concept shaped governments. The Enlightenment promoted religious tolerance and free inquiry. These ideas remain central to liberal democracy."
        },
        {
            "id": "history_industrial_revolution",
            "text": "The Industrial Revolution began in Britain in the late 18th century. Steam power mechanized manufacturing. Factories replaced cottage industries. Railroads transformed transportation. Urbanization accelerated as workers moved to cities. Living conditions initially worsened before reforms improved them. Industrialization created modern economic systems."
        },
        {
            "id": "history_american_revolution",
            "text": "The American Revolution established the United States as an independent nation. Colonists protested taxation without representation. The Declaration of Independence proclaimed natural rights and self-governance. George Washington led the Continental Army to victory. The Constitution created a federal republic with separated powers. The revolution inspired independence movements worldwide."
        },
        {
            "id": "history_french_revolution",
            "text": "The French Revolution overthrew the monarchy and transformed society. Economic crisis and inequality fueled discontent. The storming of the Bastille symbolized revolution. The Declaration of the Rights of Man proclaimed liberty and equality. The Reign of Terror executed thousands. Napoleon eventually seized power, spreading revolutionary ideas through conquest."
        },
        {
            "id": "history_ww1",
            "text": "World War One occurred between 1914 and 1918. The assassination of Archduke Franz Ferdinand triggered the conflict. Alliance systems drew major powers into war. Trench warfare caused massive casualties with little territorial gain. New weapons included machine guns, poison gas, and tanks. Approximately sixteen million died. The Treaty of Versailles imposed harsh terms on Germany."
        },
        {
            "id": "history_ww2",
            "text": "World War Two occurred between 1939 and 1945. Nazi Germany's invasion of Poland triggered declarations of war. The Holocaust murdered six million Jews and millions of others. The war spread across Europe, Africa, Asia, and the Pacific. Approximately seventy million people died. Atomic bombs on Hiroshima and Nagasaki ended the war. The United Nations formed to prevent future conflicts."
        },
        {
            "id": "history_cold_war",
            "text": "The Cold War was ideological conflict between the United States and Soviet Union from 1947 to 1991. Nuclear weapons created mutual assured destruction. Proxy wars occurred in Korea, Vietnam, and elsewhere. The space race demonstrated technological competition. The Berlin Wall symbolized division. The Soviet Union dissolved in 1991, ending the Cold War."
        },
        {
            "id": "history_civil_rights",
            "text": "The Civil Rights Movement fought for racial equality in America. Segregation laws enforced separation and discrimination. Rosa Parks' bus boycott sparked protests. Martin Luther King Jr. led nonviolent resistance. The Civil Rights Act of 1964 outlawed discrimination. The Voting Rights Act of 1965 protected voting rights. The movement transformed American society."
        },
        {
            "id": "history_digital_revolution",
            "text": "The Digital Revolution transformed society through computing and the internet. Personal computers became widespread in the 1980s. The World Wide Web launched in 1991. Mobile phones evolved into smartphones. Social media connected billions. Digital technology changed communication, commerce, and culture. We are still experiencing this ongoing transformation."
        },
        {
            "id": "history_internet",
            "text": "The internet originated from ARPANET, a military research network in the 1960s. TCP/IP protocols enabled network interconnection. Tim Berners-Lee invented the World Wide Web in 1989. Browsers made the web accessible. The dot-com boom commercialized the internet. Today, the internet connects billions of people and devices worldwide."
        },
    ],

    # =========================================================================
    # CYBERSECURITY - EXPANDED
    # =========================================================================
    "cybersecurity": [
        {
            "id": "cyber_intro",
            "text": "Cybersecurity protects systems, networks, and data from digital attacks. The CIA triad represents confidentiality, integrity, and availability. Confidentiality prevents unauthorized access. Integrity ensures data accuracy. Availability maintains system access. Security is a continuous process, not a one-time implementation."
        },
        {
            "id": "cyber_threats",
            "text": "Cyber threats come from various actors with different motivations. Nation-states conduct espionage and sabotage. Criminals seek financial gain through ransomware and fraud. Hacktivists pursue political goals. Insiders abuse authorized access. Understanding threat actors helps prioritize defenses."
        },
        {
            "id": "cyber_encryption_symmetric",
            "text": "Symmetric encryption uses the same key for encryption and decryption. AES is the current standard symmetric algorithm. Key distribution is the main challenge since both parties need the key. Symmetric encryption is fast and suitable for large data. It protects data at rest and in transit."
        },
        {
            "id": "cyber_encryption_asymmetric",
            "text": "Asymmetric encryption uses public and private key pairs. The public key encrypts; only the private key decrypts. RSA and elliptic curve cryptography are common algorithms. Asymmetric encryption solves key distribution but is slower. It enables digital signatures and secure key exchange."
        },
        {
            "id": "cyber_hashing",
            "text": "Hashing converts data to fixed-size fingerprints. Hash functions are one-way; you cannot reverse them. The same input always produces the same hash. Different inputs should produce different hashes. Hashing verifies data integrity and stores passwords securely. SHA-256 is the current standard."
        },
        {
            "id": "cyber_passwords",
            "text": "Password security requires proper storage and policies. Never store passwords in plain text. Hash passwords with salt to prevent rainbow table attacks. Use bcrypt, scrypt, or Argon2 for password hashing. Enforce minimum length and complexity. Enable multi-factor authentication for additional security."
        },
        {
            "id": "cyber_phishing",
            "text": "Phishing tricks users into revealing sensitive information. Attackers impersonate trusted entities through email or websites. Spear phishing targets specific individuals with personalized attacks. Check sender addresses and URLs carefully. Never enter credentials on suspicious sites. Report phishing attempts to security teams."
        },
        {
            "id": "cyber_malware",
            "text": "Malware is malicious software designed to harm systems. Viruses attach to programs and spread when executed. Worms spread automatically across networks. Trojans disguise as legitimate software. Ransomware encrypts files and demands payment. Keep software updated and use antivirus protection."
        },
        {
            "id": "cyber_injection",
            "text": "Injection attacks insert malicious code into applications. SQL injection manipulates database queries through user input. Command injection executes system commands. Cross-site scripting injects JavaScript into web pages. Prevent injection by validating input and using parameterized queries. Never trust user input."
        },
        {
            "id": "cyber_network_security",
            "text": "Network security protects data in transit and network infrastructure. Firewalls filter traffic based on rules. Intrusion detection systems monitor for suspicious activity. VPNs encrypt connections over public networks. Network segmentation limits breach impact. Defense in depth uses multiple security layers."
        },
        {
            "id": "cyber_authentication",
            "text": "Authentication verifies user identity. Something you know includes passwords and PINs. Something you have includes phones and security keys. Something you are includes fingerprints and face recognition. Multi-factor authentication combines multiple factors. Strong authentication prevents unauthorized access."
        },
        {
            "id": "cyber_authorization",
            "text": "Authorization controls what authenticated users can access. The principle of least privilege grants minimum necessary access. Role-based access control assigns permissions to roles. Access control lists specify permissions for resources. Regular access reviews remove unnecessary permissions. Authorization limits damage from compromised accounts."
        },
        {
            "id": "cyber_incident_response",
            "text": "Incident response handles security breaches systematically. Preparation establishes procedures and teams. Detection identifies potential incidents. Containment limits damage. Eradication removes threats. Recovery restores normal operations. Lessons learned improve future response. Practice incident response through tabletop exercises."
        },
        {
            "id": "cyber_social_engineering",
            "text": "Social engineering manipulates people to bypass security. Pretexting creates false scenarios to gain trust. Baiting offers something enticing to deliver malware. Tailgating follows authorized people into secure areas. Security awareness training helps employees recognize attacks. Humans are often the weakest security link."
        },
        {
            "id": "cyber_secure_development",
            "text": "Secure development integrates security throughout the software lifecycle. Threat modeling identifies potential vulnerabilities early. Secure coding practices prevent common vulnerabilities. Code review catches security issues. Security testing validates defenses. Fixing vulnerabilities early costs less than fixing them in production."
        },
    ],

    # =========================================================================
    # DATABASES - NEW CATEGORY
    # =========================================================================
    "databases": [
        {
            "id": "db_intro",
            "text": "Databases store, organize, and retrieve data efficiently. They provide persistence beyond program execution. Database management systems handle concurrent access, transactions, and recovery. Choosing the right database depends on data structure, scale, and access patterns. Databases are fundamental to nearly all applications."
        },
        {
            "id": "db_relational_intro",
            "text": "Relational databases organize data into tables with rows and columns. Tables relate through foreign keys. SQL queries retrieve and manipulate data. ACID properties ensure transaction reliability. PostgreSQL, MySQL, and SQLite are popular relational databases. Relational databases excel for structured data with complex relationships."
        },
        {
            "id": "db_sql_basics",
            "text": "SQL is the standard language for relational databases. SELECT retrieves data from tables. INSERT adds new rows. UPDATE modifies existing rows. DELETE removes rows. WHERE clauses filter results. JOIN combines data from multiple tables. SQL is declarative, specifying what data you want, not how to get it."
        },
        {
            "id": "db_sql_joins",
            "text": "Joins combine rows from multiple tables based on related columns. Inner join returns only matching rows. Left join includes all rows from the left table. Right join includes all rows from the right table. Full outer join includes all rows from both tables. Understanding joins is essential for relational database queries."
        },
        {
            "id": "db_sql_indexes",
            "text": "Indexes speed up data retrieval by creating sorted references. Without indexes, queries scan entire tables. B-tree indexes handle equality and range queries. Hash indexes optimize equality comparisons. Indexes slow down writes and use storage. Index columns used frequently in WHERE clauses and joins."
        },
        {
            "id": "db_normalization",
            "text": "Normalization organizes data to reduce redundancy and improve integrity. First normal form eliminates repeating groups. Second normal form removes partial dependencies. Third normal form removes transitive dependencies. Normalization prevents update anomalies. Denormalization trades redundancy for query performance when needed."
        },
        {
            "id": "db_transactions",
            "text": "Transactions group operations into atomic units. ACID properties ensure reliability. Atomicity means all operations succeed or all fail. Consistency maintains database rules. Isolation prevents concurrent transaction interference. Durability ensures committed changes persist. Transactions protect data integrity during failures."
        },
        {
            "id": "db_nosql_intro",
            "text": "NoSQL databases provide alternatives to relational models. Document databases store JSON-like documents. Key-value stores provide simple lookup by key. Column-family databases optimize for wide tables. Graph databases model relationships. NoSQL databases often sacrifice consistency for scalability and flexibility."
        },
        {
            "id": "db_mongodb",
            "text": "MongoDB is a popular document database storing JSON-like documents. Documents can have varying structures within collections. Queries use a rich expression language. Aggregation pipelines process data in stages. MongoDB scales horizontally through sharding. It excels for applications with evolving schemas and document-oriented data."
        },
        {
            "id": "db_redis",
            "text": "Redis is an in-memory data store supporting various data structures. It provides strings, lists, sets, sorted sets, and hashes. Redis excels as a cache, reducing database load. Pub/sub enables real-time messaging. Redis is extremely fast but limited by memory. It's ideal for caching, sessions, and real-time features."
        },
        {
            "id": "db_postgresql",
            "text": "PostgreSQL is a powerful open-source relational database. It supports advanced features like JSON, full-text search, and geospatial data. PostgreSQL emphasizes standards compliance and extensibility. It handles complex queries and large datasets well. PostgreSQL is often the best choice for general-purpose relational needs."
        },
        {
            "id": "db_scaling",
            "text": "Database scaling handles growing data and traffic. Vertical scaling adds resources to a single server. Horizontal scaling distributes data across multiple servers. Replication copies data for redundancy and read scaling. Sharding partitions data across servers. Caching reduces database load. Scaling strategies depend on workload characteristics."
        },
    ],

    # =========================================================================
    # CLOUD COMPUTING - NEW CATEGORY
    # =========================================================================
    "cloud_computing": [
        {
            "id": "cloud_intro",
            "text": "Cloud computing provides on-demand computing resources over the internet. Instead of owning hardware, you rent capacity from providers. Benefits include scalability, flexibility, and reduced capital expenses. Major providers are Amazon Web Services, Microsoft Azure, and Google Cloud. Cloud computing transformed how organizations deploy and manage technology."
        },
        {
            "id": "cloud_iaas",
            "text": "Infrastructure as a Service provides virtualized computing resources. You rent virtual machines, storage, and networking. You manage operating systems and applications. IaaS offers maximum flexibility and control. Examples include EC2, Azure Virtual Machines, and Google Compute Engine. IaaS suits workloads requiring specific configurations."
        },
        {
            "id": "cloud_paas",
            "text": "Platform as a Service provides managed application platforms. The provider handles infrastructure, operating systems, and runtime. You deploy and manage applications. PaaS simplifies development and operations. Examples include Heroku, Google App Engine, and Azure App Service. PaaS accelerates development by removing infrastructure concerns."
        },
        {
            "id": "cloud_saas",
            "text": "Software as a Service delivers applications over the internet. Users access software through browsers without installation. The provider manages everything from infrastructure to application. Examples include Gmail, Salesforce, and Microsoft 365. SaaS eliminates software maintenance for users."
        },
        {
            "id": "cloud_serverless",
            "text": "Serverless computing runs code without managing servers. Functions execute in response to events. You pay only for actual execution time. The platform handles scaling automatically. AWS Lambda, Azure Functions, and Google Cloud Functions are examples. Serverless simplifies deployment and reduces operational overhead."
        },
        {
            "id": "cloud_containers",
            "text": "Containers package applications with their dependencies for consistent deployment. Docker is the standard container runtime. Containers are lighter than virtual machines, sharing the host kernel. Container images ensure identical environments across development and production. Containers enable microservices architectures."
        },
        {
            "id": "cloud_kubernetes",
            "text": "Kubernetes orchestrates container deployment, scaling, and management. It schedules containers across clusters of machines. Services provide stable networking for pods. Deployments manage application updates. Kubernetes has become the standard for container orchestration. It enables reliable, scalable container deployments."
        },
        {
            "id": "cloud_storage",
            "text": "Cloud storage provides scalable, durable data storage. Object storage like S3 stores unstructured data at scale. Block storage provides persistent volumes for virtual machines. File storage offers shared file systems. Cloud storage eliminates capacity planning and provides high durability."
        },
        {
            "id": "cloud_networking",
            "text": "Cloud networking connects resources securely. Virtual private clouds isolate your resources. Subnets segment networks. Security groups control traffic. Load balancers distribute requests across instances. Content delivery networks cache content globally. Cloud networking provides flexibility with enterprise-grade capabilities."
        },
        {
            "id": "cloud_security",
            "text": "Cloud security is a shared responsibility between provider and customer. Providers secure infrastructure. Customers secure their data and access. Identity and access management controls permissions. Encryption protects data at rest and in transit. Security monitoring detects threats. Understand your security responsibilities in the cloud."
        },
    ],

    # =========================================================================
    # OPERATING SYSTEMS - NEW CATEGORY
    # =========================================================================
    "operating_systems": [
        {
            "id": "os_intro",
            "text": "Operating systems manage computer hardware and provide services for applications. They handle process scheduling, memory management, file systems, and device drivers. Major operating systems include Windows, macOS, Linux, Android, and iOS. Understanding operating systems helps write efficient software and troubleshoot problems."
        },
        {
            "id": "os_processes",
            "text": "Processes are running instances of programs. Each process has its own memory space and resources. The operating system schedules processes on CPU cores. Context switching saves and restores process state. Process creation, termination, and communication are fundamental operations. Processes enable multitasking."
        },
        {
            "id": "os_threads",
            "text": "Threads are lightweight execution units within processes. Threads share process memory and resources. Multiple threads enable concurrent execution within a program. Thread synchronization prevents race conditions. Threads are cheaper to create than processes. Modern applications use threads for responsiveness and parallelism."
        },
        {
            "id": "os_memory",
            "text": "Memory management allocates and tracks system memory. Virtual memory provides each process its own address space. Paging divides memory into fixed-size blocks. The operating system swaps pages between RAM and disk. Memory protection prevents processes from accessing each other's memory. Efficient memory management is crucial for performance."
        },
        {
            "id": "os_filesystem",
            "text": "File systems organize data on storage devices. Files contain data; directories organize files hierarchically. File systems track file locations, permissions, and metadata. Common file systems include NTFS, ext4, and APFS. File systems provide abstraction over physical storage. Understanding file systems helps with data management and recovery."
        },
        {
            "id": "os_linux_basics",
            "text": "Linux is an open-source operating system kernel. Distributions like Ubuntu, Fedora, and Debian package Linux with software. The command line provides powerful system control. Linux dominates servers, cloud computing, and embedded systems. Learning Linux is essential for developers and system administrators."
        },
        {
            "id": "os_linux_commands",
            "text": "Linux commands control the system from the terminal. ls lists directory contents. cd changes directories. cp copies files. mv moves or renames files. rm removes files. grep searches text. chmod changes permissions. These commands form the foundation of Linux administration."
        },
        {
            "id": "os_permissions",
            "text": "File permissions control access to files and directories. Read, write, and execute permissions apply to owner, group, and others. Permission bits are represented numerically or symbolically. The principle of least privilege grants minimum necessary access. Proper permissions protect sensitive data and system integrity."
        },
        {
            "id": "os_shell",
            "text": "Shells provide command-line interfaces to operating systems. Bash is the most common Linux shell. Shells execute commands, run scripts, and manage processes. Environment variables configure shell behavior. Shell scripting automates repetitive tasks. Mastering the shell dramatically increases productivity."
        },
        {
            "id": "os_networking",
            "text": "Operating systems provide networking capabilities. TCP/IP protocols enable internet communication. Sockets provide programming interfaces for network communication. DNS resolves domain names to IP addresses. Firewalls filter network traffic. Understanding networking is essential for distributed applications."
        },
    ],

    # =========================================================================
    # VERSION CONTROL - NEW CATEGORY
    # =========================================================================
    "version_control": [
        {
            "id": "git_intro",
            "text": "Version control tracks changes to files over time. It enables collaboration, history tracking, and reverting mistakes. Git is the dominant version control system. Repositories store project files and history. Commits record snapshots of changes. Version control is essential for professional software development."
        },
        {
            "id": "git_basics",
            "text": "Git tracks changes in a local repository. git init creates a new repository. git add stages changes for commit. git commit records staged changes with a message. git status shows current state. git log displays commit history. These commands form the basic Git workflow."
        },
        {
            "id": "git_branching",
            "text": "Branches enable parallel development. The main branch contains stable code. Feature branches isolate new development. git branch creates branches. git checkout switches branches. git merge combines branches. Branching enables experimentation without affecting stable code."
        },
        {
            "id": "git_merging",
            "text": "Merging combines changes from different branches. Fast-forward merges move the branch pointer. Three-way merges create merge commits. Merge conflicts occur when changes overlap. Resolve conflicts by editing files and committing. Clean merges require good branch management."
        },
        {
            "id": "git_remote",
            "text": "Remote repositories enable collaboration. GitHub, GitLab, and Bitbucket host remote repositories. git clone copies remote repositories locally. git push uploads local commits. git pull downloads remote changes. git fetch retrieves without merging. Remote repositories are the collaboration hub."
        },
        {
            "id": "git_workflow",
            "text": "Git workflows organize team collaboration. Feature branch workflow creates branches for each feature. Gitflow defines branches for features, releases, and hotfixes. Trunk-based development uses short-lived branches. Pull requests enable code review before merging. Choose workflows matching team size and release cadence."
        },
        {
            "id": "git_best_practices",
            "text": "Good Git practices improve collaboration. Write clear, descriptive commit messages. Make small, focused commits. Pull frequently to stay synchronized. Review code before merging. Never commit secrets or credentials. Use gitignore to exclude generated files. Good practices prevent problems and ease collaboration."
        },
    ],

    # =========================================================================
    # SOFT SKILLS - NEW CATEGORY
    # =========================================================================
    "soft_skills": [
        {
            "id": "skills_problem_solving",
            "text": "Problem solving is the core skill of engineering. Define the problem clearly before seeking solutions. Break complex problems into smaller parts. Consider multiple approaches before committing. Test solutions incrementally. Learn from failures. Systematic problem solving produces better results than random attempts."
        },
        {
            "id": "skills_debugging",
            "text": "Debugging finds and fixes software defects. Reproduce the problem consistently first. Form hypotheses about causes. Test hypotheses systematically. Use debuggers to inspect program state. Add logging to trace execution. Rubber duck debugging explains code to find issues. Patience and method solve bugs."
        },
        {
            "id": "skills_communication",
            "text": "Communication skills are essential for engineers. Explain technical concepts to non-technical audiences. Write clear documentation. Participate constructively in code reviews. Ask questions when uncertain. Share knowledge with teammates. Good communication multiplies individual effectiveness."
        },
        {
            "id": "skills_learning",
            "text": "Continuous learning is necessary in technology. The field evolves rapidly. Read documentation and tutorials. Build projects to apply knowledge. Learn from code reviews and feedback. Teach others to deepen understanding. Embrace being a beginner in new areas. Learning is a career-long practice."
        },
        {
            "id": "skills_time_management",
            "text": "Time management maximizes productivity. Prioritize tasks by importance and urgency. Break large tasks into smaller steps. Estimate time realistically. Minimize context switching. Protect focused work time. Review and adjust plans regularly. Good time management reduces stress and improves output."
        },
        {
            "id": "skills_collaboration",
            "text": "Collaboration multiplies team effectiveness. Share information openly. Give and receive feedback constructively. Respect different perspectives and approaches. Help teammates when they struggle. Celebrate team successes. Conflict is normal; resolve it professionally. Strong teams outperform strong individuals."
        },
    ],
}


# =============================================================================
# SLEEP LEARNING GENERATOR CLASS
# =============================================================================

class SleepLearningGenerator:
    """
    Generates educational audio using JARVIS voice.
    Uses SAME settings as jarvis_voice.py for consistency.
    
    Each item = separate audio file (30 sec to 2 min)
    Files are NOT linked - play in any order or shuffle!
    """
    
    def __init__(self, output_dir: str = "./jarvis_sleep_learning"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.xtts = None
        self.voice_sample = None
        self.sample_rate = TTS_SETTINGS["sample_rate"]
        
        self._init_tts()
    
    def _init_tts(self):
        """Initialize TTS with SAME settings as jarvis_voice.py"""
        if not COQUI_AVAILABLE:
            print("[ERROR] Coqui TTS not available!")
            return
        
        print("[Generator] Loading XTTS with STABLE settings...")
        
        try:
            self.xtts = TTS(TTS_SETTINGS["model"])
            
            # Apply PHONEME LOOP FIX settings (SAME as jarvis_voice.py)
            try:
                if hasattr(self.xtts, 'synthesizer') and self.xtts.synthesizer:
                    synth = self.xtts.synthesizer
                    if hasattr(synth, 'args'):
                        synth.args.use_deterministic_seed = True
                        synth.args.denoise_audio = False  # CRITICAL - prevents loops
                        synth.args.encoder_temperature = TTS_SETTINGS["encoder_temperature"]
                    
                    tts_model = synth.tts_model
                    if tts_model and hasattr(tts_model, 'config'):
                        tts_model.config.temperature = TTS_SETTINGS["temperature"]
                        tts_model.config.repetition_penalty = TTS_SETTINGS["repetition_penalty"]
                        tts_model.config.top_k = TTS_SETTINGS["top_k"]
                        tts_model.config.top_p = TTS_SETTINGS["top_p"]
                    print("[Generator] ✓ Applied PHONEME LOOP FIX config")
            except Exception as e:
                print(f"[Generator] Config tweak skipped: {e}")
            
            print("[Generator] ✓ XTTS ready")
        except Exception as e:
            print(f"[Generator] XTTS failed: {e}")
            return
        
        # Find voice sample
        for sample_name in VOICE_SAMPLES:
            sample_path = Path(sample_name)
            if sample_path.exists():
                self.voice_sample = str(sample_path)
                print(f"[Generator] ✓ Voice sample: {self.voice_sample}")
                break
            # Also check parent directory
            parent_path = Path("..") / sample_name
            if parent_path.exists():
                self.voice_sample = str(parent_path)
                print(f"[Generator] ✓ Voice sample: {self.voice_sample}")
                break
        
        if not self.voice_sample:
            print("[ERROR] No voice sample found!")
            print("Looking for:", VOICE_SAMPLES)
    
    def _split_text_into_chunks(self, text: str, max_chars: int = 240) -> list:
        """
        Split long text into chunks at sentence boundaries.
        XTTS can handle ~250 chars, we use 240 to be safe while minimizing splits.
        """
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        # Split by periods, question marks, exclamation marks
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If single sentence is too long, split by commas
            if len(sentence) > max_chars:
                comma_parts = sentence.split(', ')
                for part in comma_parts:
                    part = part.strip()
                    if len(current_chunk) + len(part) + 2 > max_chars:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = part
                    else:
                        if current_chunk:
                            current_chunk = current_chunk + ", " + part
                        else:
                            current_chunk = part
                continue
            
            # If adding this sentence exceeds limit, save current chunk
            if len(current_chunk) + len(sentence) + 1 > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = (current_chunk + " " + sentence).strip()
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Final safety check - force split any chunk still too long
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > max_chars:
                # Hard split at max_chars
                for i in range(0, len(chunk), max_chars):
                    final_chunks.append(chunk[i:i+max_chars])
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def generate_audio(self, text: str) -> Optional[np.ndarray]:
        """Generate audio with SAME settings as jarvis_voice.py. Handles long text by chunking."""
        if not self.xtts or not self.voice_sample:
            return None
        
        try:
            # Split long text into chunks to avoid truncation
            chunks = self._split_text_into_chunks(text)
            
            if len(chunks) == 1:
                # Short text - generate directly
                audio = self.xtts.tts(
                    text=text,
                    speaker_wav=self.voice_sample,
                    language=TTS_SETTINGS["language"],
                    speed=TTS_SETTINGS["speed"],
                    split_sentences=TTS_SETTINGS["split_sentences"],
                )
                return np.array(audio, dtype=np.float32)
            else:
                # Long text - generate chunks and concatenate
                print(f"    [Chunking: {len(chunks)} parts]")
                audio_parts = []
                
                # Import torch for memory cleanup
                try:
                    import torch
                    has_torch = True
                except ImportError:
                    has_torch = False
                
                for i, chunk in enumerate(chunks):
                    print(f"      Part {i+1}/{len(chunks)}: {chunk[:50]}...")
                    chunk_audio = self.xtts.tts(
                        text=chunk,
                        speaker_wav=self.voice_sample,
                        language=TTS_SETTINGS["language"],
                        speed=TTS_SETTINGS["speed"],
                        split_sentences=TTS_SETTINGS["split_sentences"],
                    )
                    audio_parts.append(np.array(chunk_audio, dtype=np.float32))
                    
                    # Tiny pause between chunks - just 50ms for natural flow
                    # Only add pause if not the last chunk
                    if i < len(chunks) - 1:
                        pause = np.zeros(int(self.sample_rate * 0.05), dtype=np.float32)
                        audio_parts.append(pause)
                    
                    # Clear GPU memory between chunks
                    if has_torch and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Concatenate all parts
                return np.concatenate(audio_parts)
                
        except Exception as e:
            print(f"[Generator] Error: {e}")
            return None
    
    def generate_category(self, category: str) -> int:
        """Generate all audio for a category"""
        if category not in SLEEP_CONTENT:
            print(f"[Error] Unknown category: {category}")
            print(f"Available: {list(SLEEP_CONTENT.keys())}")
            return 0
        
        items = SLEEP_CONTENT[category]
        category_dir = self.output_dir / category
        category_dir.mkdir(exist_ok=True)
        
        generated = 0
        total = len(items)
        
        print(f"\n{'='*60}")
        print(f"Generating: {category} ({total} items)")
        print(f"{'='*60}\n")
        
        for i, item in enumerate(items):
            item_id = item["id"]
            text = item["text"]
            output_path = category_dir / f"{item_id}.wav"
            
            # Skip if already exists
            if output_path.exists():
                print(f"[{i+1}/{total}] ✓ {item_id} (cached)")
                generated += 1
                continue
            
            print(f"[{i+1}/{total}] Generating: {item_id}")
            print(f"    Text: {text[:60]}...")
            
            start_time = time.time()
            audio = self.generate_audio(text)
            gen_time = time.time() - start_time
            
            if audio is not None:
                # Save as WAV
                import scipy.io.wavfile as wav
                wav.write(str(output_path), self.sample_rate, audio)
                
                # Calculate duration
                duration = len(audio) / self.sample_rate
                print(f"[{i+1}/{total}] ✓ Saved: {output_path.name} ({duration:.1f}s audio, {gen_time:.1f}s to generate)")
                generated += 1
            else:
                print(f"[{i+1}/{total}] ✗ Failed: {item_id}")
        
        return generated
    
    def generate_all(self):
        """Generate all categories"""
        total_generated = 0
        total_items = sum(len(items) for items in SLEEP_CONTENT.values())
        
        print("\n" + "="*60)
        print("🌙 JARVIS Sleep Learning Generator")
        print("="*60)
        print(f"📊 Total items: {total_items}")
        print(f"📁 Output: {self.output_dir}")
        print(f"🎯 Each item = separate audio file")
        print(f"⏱️ Estimated time: {total_items * 1.5 / 60:.1f} - {total_items * 3 / 60:.1f} hours")
        print("="*60 + "\n")
        
        start_time = time.time()
        
        for category in SLEEP_CONTENT.keys():
            generated = self.generate_category(category)
            total_generated += generated
            print(f"\n[Progress] {total_generated}/{total_items} complete\n")
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*60)
        print(f"✅ Complete! Generated {total_generated}/{total_items} files")
        print(f"⏱️ Time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
        print(f"📁 Output: {self.output_dir}")
        print("="*60)
    
    def list_categories(self):
        """List all available categories with item counts"""
        total = 0
        print("\n📚 Available Categories:")
        print("-" * 50)
        for category, items in SLEEP_CONTENT.items():
            count = len(items)
            total += count
            # Estimate duration (avg ~50 words = ~20 sec)
            avg_words = sum(len(item["text"].split()) for item in items) / count
            est_duration = (avg_words / 150) * 60  # 150 words per minute
            print(f"  {category:25} {count:3} items (~{est_duration * count / 60:.0f} min)")
        print("-" * 50)
        print(f"  {'TOTAL':25} {total:3} items")
        print(f"\n  Estimated total audio: ~{total * 0.75:.0f} - {total * 1.5:.0f} minutes")
        print(f"  Estimated generation time: ~{total * 1.5 / 60:.1f} - {total * 3 / 60:.1f} hours")
    
    def estimate(self):
        """Estimate generation time and file sizes"""
        total_items = sum(len(items) for items in SLEEP_CONTENT.values())
        total_words = sum(
            len(item["text"].split()) 
            for items in SLEEP_CONTENT.values() 
            for item in items
        )
        
        # Estimates based on typical TTS performance
        avg_words_per_item = total_words / total_items
        est_audio_per_item = avg_words_per_item / 150 * 60  # seconds (150 wpm)
        est_gen_time_per_item = 90  # seconds average generation time
        
        total_audio_minutes = (est_audio_per_item * total_items) / 60
        total_gen_hours = (est_gen_time_per_item * total_items) / 3600
        total_size_mb = total_items * 3  # ~3MB per file average
        
        print("\n" + "="*60)
        print("📊 GENERATION ESTIMATES")
        print("="*60)
        print(f"  Total items:           {total_items}")
        print(f"  Total words:           {total_words:,}")
        print(f"  Avg words per item:    {avg_words_per_item:.0f}")
        print("-"*60)
        print(f"  Est. audio per item:   ~{est_audio_per_item:.0f} seconds")
        print(f"  Est. total audio:      ~{total_audio_minutes:.0f} minutes ({total_audio_minutes/60:.1f} hours)")
        print("-"*60)
        print(f"  Est. generation time:  ~{total_gen_hours:.1f} - {total_gen_hours * 1.5:.1f} hours")
        print(f"  Est. total size:       ~{total_size_mb:.0f} - {total_size_mb * 1.5:.0f} MB")
        print("="*60)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("🤖 JARVIS Sleep Learning Voice Generator")
    print("   Run before bed - wake up smarter!")
    print("   Each topic = separate audio file")
    print("="*60)
    
    if not COQUI_AVAILABLE:
        print("\n[ERROR] Coqui TTS not installed!")
        print("Run: pip install TTS")
        sys.exit(1)
    
    generator = SleepLearningGenerator()
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg == "--list":
            generator.list_categories()
        elif arg == "--estimate":
            generator.estimate()
        elif arg == "--category" and len(sys.argv) > 2:
            generator.generate_category(sys.argv[2])
        elif arg == "--help":
            print("\nUsage:")
            print("  python generate_sleep_learning.py              # Generate all")
            print("  python generate_sleep_learning.py --list       # List categories")
            print("  python generate_sleep_learning.py --estimate   # Show estimates")
            print("  python generate_sleep_learning.py --category python_libraries")
        else:
            print(f"Unknown option: {arg}")
            print("Use --help for usage information")
    else:
        generator.generate_all()
