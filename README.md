# Product Categorization Service 🏷️

## Overview

This project demonstrates an end-to-end Machine Learning solution for automatically categorizing building products based on their name. Built with modern MLOps practices, it showcases how to deploy a production-ready ML service that can handle real-time categorization requests while maintaining model performance tracking and easy retraining capabilities.

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-009688.svg)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-latest-0194E2.svg)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-latest-2496ED.svg)](https://www.docker.com)

## 🎯 Business Problem

Hardware stores often struggle with:
- Manually categorizing thousands of products
- Maintaining consistency in product categorization
- Efficiently organizing inventory
- Processing new product entries quickly

This solution automates the categorization process, reducing manual effort and ensuring consistent categorization across the product catalog.

## 🔍 Solution Architecture

The project implements a modular architecture with three main components:

### 1. Training Pipeline 🚂
- **Data Preprocessing**: 
  - Text cleaning and normalization
  - Stopword removal (Spanish & English)
  - Lemmatization for better feature extraction
- **Model Training**:
  - Random Forest Classifier with optimized hyperparameters
  - Cross-validation for model selection
  - Feature engineering using CountVectorizer
- **MLflow Integration**:
  - Automatic experiment tracking
  - Model versioning and registry
  - Performance metrics logging
  - Easy model deployment management

### 2. Inference Service 🎯
- Real-time prediction endpoint
- Batch prediction capabilities
- Model version control
- Input validation and error handling
- Scalable architecture

### 3. API Layer 🌐
Built with FastAPI, offering:
- Fast and efficient request handling
- Automatic API documentation (OpenAPI/Swagger)
- Input/Output validation with Pydantic
- Async support for better performance
- Health check endpoints
- Background task processing for training jobs

## ⚡ Challenges Overcome

- Model Versioning (MLflow): Implemented automatic comparison between retrained and production models using the same test set. Only models with better F1-score are promoted as challengers for potential production.
- Robust Validation (Pydantic): Ensured reliable request/response handling through strict schema validation, reducing errors and improving API consistency.
- Async Retraining Pipeline: Designed retraining as an asynchronous background task, keeping the API responsive even during long training processes.

## 📊 Model Performance

The current model achieves:
- Accuracy: ~90% across all categories
- Macro F1-Score: 0.89
- Fast inference time: <100ms per request

## 🔄 API Endpoints

### Training
```http
GET /training
```
Initiates a new training job (runs asynchronously)

### Inference
```http
POST /v1/categorizer/inference
```
Get category predictions for product names

### Health Check
```http
GET /health
```
Service health status

## 💡 Technical Highlights

1. **MLOps Best Practices**
   - Experiment tracking with MLflow
   - Model versioning and registry
   - Containerized deployment
   - Automated training pipeline

2. **Scalable Architecture**
   - Asynchronous training jobs
   - Background task processing
   - Modular code structure
   - Easy to extend and maintain

3. **Production-Ready Features**
   - Error handling and logging
   - Input validation
   - Performance monitoring
   - Docker support

## 🛠️ Technologies Used

- **FastAPI**: Modern, fast web framework for building APIs
- **MLflow**: ML lifecycle management
- **scikit-learn**: Machine learning algorithms
- **NLTK**: Natural Language Processing
- **Docker**: Containerization
- **Pydantic**: Data validation
- **Uvicorn**: ASGI server
- **Flake8**: Code style and standards enforcement  

## 📈 Future Improvements

- [ ] Add support for more product categories
- [ ] Implement A/B testing framework
- [ ] Add model retraining triggers based on performance metrics
- [ ] Implement data drift detection

## ✨ Author

**Ivan Condori**
- GitHub: [@icondor2019](https://github.com/icondor2019)
