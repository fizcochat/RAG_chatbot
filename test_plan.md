# Fiscozen Chatbot Test Plan

This document outlines the testing procedures to verify the successful integration of all features from both chatbot versions.

## 1. Basic Functionality Tests

### 1.1 Application Launch
- [ ] Test launching app with `streamlit run app.py`
- [ ] Test launching app with `python run_fiscozen.py`
- [ ] Verify that the app properly initializes and loads all components

### 1.2 UI Verification
- [ ] Check that Fiscozen logo and branding are displayed correctly
- [ ] Verify that the chat interface is functional and visually consistent
- [ ] Confirm sidebar elements are displayed correctly

## 2. Multilingual Support Tests

### 2.1 Language Selection
- [ ] Verify that language selector in sidebar works correctly
- [ ] Confirm that UI elements update when switching languages
- [ ] Check that language preference is maintained during the session

### 2.2 Italian Mode
- [ ] Test answering tax-related questions in Italian
- [ ] Verify that responses are returned in Italian
- [ ] Ensure suggested questions are displayed in Italian

### 2.3 English Mode
- [ ] Test answering tax-related questions in English
- [ ] Verify that questions are properly translated to Italian internally
- [ ] Confirm that Italian responses are translated back to English
- [ ] Ensure suggested questions are displayed in English

## 3. Topic Filtering Tests

### 3.1 FastText Model
- [ ] Verify that FastText model loads correctly
- [ ] Test with tax-related queries and confirm they're accepted
- [ ] Test with non-tax queries and confirm they're properly identified as out-of-scope
- [ ] Check appropriate messages for out-of-scope queries in both languages

## 4. Monitoring Functionality Tests

### 4.1 Dashboard Access
- [ ] Access monitoring dashboard via URL parameter `?page=monitor`
- [ ] Verify password protection works correctly
- [ ] Check that dashboard loads with proper statistics and graphs

### 4.2 Logging
- [ ] Verify that successful queries are logged with proper details
- [ ] Confirm that out-of-scope queries are logged
- [ ] Test feedback functionality and ensure feedback is logged
- [ ] Check that performance metrics (response time) are captured

## 5. RAG System Tests

### 5.1 Query Processing
- [ ] Verify that queries are properly refined using context
- [ ] Test that relevant documents are retrieved from Pinecone
- [ ] Confirm that responses are generated based on retrieved content

### 5.2 Context Management
- [ ] Test multi-turn conversations to verify context is maintained
- [ ] Confirm that previous conversation history influences responses

## 6. Edge Cases and Error Handling

### 6.1 Error Handling
- [ ] Test with missing API keys
- [ ] Verify app behavior when translation services fail
- [ ] Check error handling when Pinecone is unavailable
- [ ] Test with extremely long inputs

### 6.2 Windows Compatibility
- [ ] Confirm application works on Windows systems
- [ ] Verify file paths are handled correctly on Windows
- [ ] Check that monitoring features function properly on Windows

## 7. Performance and Resource Usage

### 7.1 Response Time
- [ ] Measure average response time for various query types
- [ ] Verify that multilingual processing doesn't significantly impact performance

### 7.2 Memory Usage
- [ ] Monitor memory usage during extended chat sessions
- [ ] Check for any memory leaks during prolonged usage

## Test Results and Issues

Use this section to document test results and any issues encountered:

| Test ID | Date | Result | Issues/Comments |
|---------|------|--------|-----------------|
|         |      |        |                 |
|         |      |        |                 |
|         |      |        |                 | 