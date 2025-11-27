import asyncio
import json
import os
import uuid


from toolboxv2 import App, Result, get_app
from .arXivCrawler import ArXivPDFProcessor

# Initialize module
MOD_NAME = "TruthSeeker"
version = "1.0"
export = get_app(MOD_NAME).tb

@export(mod_name=MOD_NAME, version=version, initial=True)
def initialize_module(app: App):
    """Initialize the module and register UI with CloudM"""
    # Register the UI with CloudM
    app.run_any(("CloudM", "add_ui"),
                name="TruthSeeker",
                title="TruthSeeker Research",
                path=f"/api/{MOD_NAME}/get_main_ui",
                description="AI Research Assistant"
                )

    # Initialize SSE message queues
    if not hasattr(app, 'sse_queues'):
        app.sse_queues = {}
    print("TruthSeeker online")
    return Result.ok(info="ArXivPDFProcessor UI initialized")


@export(mod_name=MOD_NAME, api=True, version=version)
async def get_main_ui(app: App):
    css= """
    <style>
    .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 1rem;
    }

    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 2rem;
    }

    .header h1 {
        font-size: 1.875rem;
        font-weight: 700;
        margin: 0;
        color: var(--text-primary);
    }

    .card {
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 0.75rem;
        box-shadow: var(--shadow);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }

    .input-group {
        margin-bottom: 1.5rem;
    }

    .input-group label {
        display: block;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }

    .input-field {
        width: 100%;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        border: 1px solid var(--theme-text-muted);
        background-color: var(--background-color);
        font-size: 1rem;
        color: var(--text-primary);
        transition: border-color 0.2s;
    }

    .input-field:focus {
        outline: none;
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
    }

    .btn {
        display: inline-block;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: 500;
        text-align: center;
        border-radius: 0.5rem;
        border: none;
        cursor: pointer;
        transition: all 0.2s;
    }

    .btn-primary {
        background-color: var(--primary);
        color: white;
    }

    .btn-primary:hover {
        transform: translateY(-1px);
    }

    .btn-secondary {
        background-color: var(--theme-accent);
        color: var(--theme-secondary);
    }

    .btn-secondary:hover {
        background-color: var(--theme-accent);
    }

    .btn-block {
        display: block;
        width: 100%;
    }

    .progress-container {
        height: 8px;
        background-color: var(--theme-primary);
        border-radius: 4px;
        margin: 1rem 0;
        overflow: hidden;
    }

    .progress-bar {
        height: 100%;
        background-color: var(--primary);
        border-radius: 4px;
        width: 0%;
        transition: width 0.5s ease;
    }



    .spinner {
        display: inline-block;
        width: 1.5rem;
        height: 1.5rem;
        vertical-align: middle;
        border: 3px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top-color: #fff;
        animation: spin 1s infinite linear;
        margin-right: 0.5rem;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    .badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.375rem;
        font-size: 0.75rem;
        font-weight: 500;
    }

    .badge-success {
        background-color: rgba(16, 185, 129, 0.1);
        color: var(--success-color);
    }

    .badge-warning {
        background-color: rgba(245, 158, 11, 0.1);
        color: var(--warning-color);
    }

    .flex {
        display: flex;
    }

    .items-center {
        align-items: center;
    }

    .justify-between {
        justify-content: space-between;
    }

    .gap-2 {
        gap: 0.5rem;
    }

    .gap-4 {
        gap: 1rem;
    }

    .mt-4 {
        margin-top: 1rem;
    }

    .ml-auto {
        margin-left: auto;
    }

    .hidden {
        display: none;
    }

    .text-sm {
        font-size: 0.875rem;
    }

    .text-center {
        text-align: center;
    }

    .font-bold {
        font-weight: 700;
    }

    .text-gray-500 {
        color: var(--theme-text-muted);
    }

    .insights-container {
        margin-top: 2rem;
    }

    .insight-card {
        border-left: 4px solid var(--primary);
        background-color: var(--card-bg);
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
    }

    .insight-card h3 {
        margin-top: 0;
        margin-bottom: 0.5rem;
        color: var(--text-primary);
    }

    .insight-card p {
        color: var(--text-primary);
        margin: 0.5rem 0;
    }

    .paper-list {
        margin-top: 1rem;
    }

    .paper-item {
        padding: 0.75rem;
        border-bottom: 1px solid var(--border-color);
    }

    .paper-item h4 {
        margin: 0 0 0.25rem 0;
    }

    .paper-item p {
        margin: 0;
        color: var(--text-primary);
        font-size: 0.875rem;
    }

    .paper-item a {
        color: var(--text-primary);
        text-decoration: none;
    }

    .donation-container {
        margin-top: 2rem;
        padding: 1.5rem;
        background-color: var(--background-color);
        border-radius: 0.75rem;
        text-align: center;
    }

    .donation-options {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }

    .donation-option {
        padding: 0.75rem 1.5rem;
        border: 1px solid var(--theme-text-muted);
        border-radius: 0.5rem;
        background-color: var(--theme-bg);
        cursor: pointer;
        transition: all 0.2s;
    }

    .donation-option:hover {
        border-color: var(--primary);
        transform: translateY(-2px);
    }

    .donation-option.selected {
        border-color: var(--primary);
        background-color: rgba(79, 70, 229, 0.1);
    }

    .donation-option input {
        width: 80px;
    }

    .tab-container {
        margin-top: 1.5rem;
    }

    .tabs {
        display: flex;
        border-bottom: 1px solid var(--border-color);
    }

    .tab {
        padding: 0.75rem 1.5rem;
        cursor: pointer;
        border-bottom: 2px solid transparent;
    }

    .tab.active {
        border-bottom-color: var(--primary);
        color: var(--primary);
        font-weight: 500;
    }

    .tab-content {
        padding: 1rem 0;
    }

    .tab-pane {
        display: none;
    }

    .tab-pane.active {
        display: block;
    }

    @media (max-width: 768px) {
        .container {
            padding: 0.5rem;
        }

        .header {
            flex-direction: column;
            text-align: center;
        }

        .header h1 {
            margin-bottom: 1rem;
        }

        .donation-options {
            flex-direction: column;
        }
    }
    </style>
    """

    # Define HTML
    html = """
    <div class="container">
        <header class="header">
            <h1>TruthSeeker Research</h1>
        </header>

        <!-- Global Revenue Progress Bar -->

        <div class="card">
            <h2>AI-Powered Academic Research</h2>
            <p>Enter your research question to search and analyze ArXiv papers using AI.</p>
            <div class="input-group">
                <label for="query">Research Question</label>
                <input type="text" id="query" class="input-field" placeholder="Enter your research question..." />
            </div>

            <div class="input-group">
                <label>Configuration</label>
                <div class="flex items-center gap-4">
                    <div>
                        <label for="maxSearch" class="text-sm">Max Search Results:</label>
                        <input type="number" id="maxSearch" min="1" max="20" value="4" class="input-field" />
                    </div>
                    <div>
                        <label for="resultsPerQuery" class="text-sm">Results Per Query:</label>
                        <input type="number" id="resultsPerQuery" min="1" max="20" value="4" class="input-field" />
                    </div>
                </div>
            </div>

            <div id="estimation-container" class="mt-4 hidden">
                <p class="text-sm">Estimated processing time: <span id="estimated-time">-</span> seconds</p>
                <p class="text-sm">Estimated cost: €<span id="estimated-cost">-</span></p>
            </div>

            <div id="progress-container" class="progress-container hidden">
                <div id="progress-bar" class="progress-bar"></div>
            </div>
            <p id="status-text" class="text-sm text-gray-500 hidden"></p>

            <div class="flex items-center gap-2 mt-4">
                <button id="search-button" class="btn btn-primary btn-block">
                    Start Research
                </button>
                <button id="stop-button" class="btn btn-secondary hidden">
                    Stop
                </button>
            </div>
        </div>

        <div id="results-container" class="hidden">
            <div class="tab-container">
                <div class="tabs">
                    <div class="tab active" data-tab="insights">Insights</div>
                    <div class="tab" data-tab="papers">Papers</div>
                    <div class="tab" data-tab="follow-up">Follow-up Questions</div>
                </div>

                <div class="tab-content">
                    <div id="insights-tab" class="tab-pane active">
                        <div id="insights-container" class="insights-container">
                            <!-- Insights will be displayed here -->
                        </div>
                    </div>

                    <div id="papers-tab" class="tab-pane">
                        <div id="papers-container" class="paper-list">
                            <!-- Papers will be displayed here -->
                        </div>
                    </div>

                    <div id="follow-up-tab" class="tab-pane">
                        <div class="card">
                            <h3>Ask a Follow-up Question</h3>
                            <div class="input-group">
                                <input type="text" id="follow-up-query" class="input-field" placeholder="Ask a question about the research..." />
                            </div>
                            <button id="follow-up-button" class="btn btn-primary btn-block">Submit Question</button>
                        </div>
                        <div id="follow-up-results" class="mt-4">
                            <!-- Follow-up results will be displayed here -->
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="donation-container">
            <h3>Support This Project</h3>
            <p>TruthSeeker Research is available on a donation basis. Your contribution helps keep this service running.</p>

            <div class="donation-options">
                <div class="donation-option selected" data-amount="2">€2</div>
                <div class="donation-option" data-amount="5">€5</div>
                <div class="donation-option" data-amount="10">€10</div>
                <div class="donation-option" data-amount="15">€15</div>
                <div class="donation-option custom">
                    <input type="number" id="custom-amount" placeholder="Custom" min="2" class="input-field" />
                </div>
            </div>

            <button id="donate-button" class="btn btn-primary mt-4">Donate</button>
        </div>


    </div>
    """

    # Define JavaScript with setTimeout and SSE instead of polling
    js = r"""
        <script unsave="true">
        setTimeout(function() {
            // Variables
            let sessionId = localStorage.getItem('researchSessionId') || generateSessionId();
            let selectedAmount = 5;
            let currentResearchId = null;
            let eventSource = null;


            // Elements
            const queryInput = document.getElementById('query');
            const searchButton = document.getElementById('search-button');
            const stopButton = document.getElementById('stop-button');
            const progressContainer = document.getElementById('progress-container');
            const progressBar = document.getElementById('progress-bar');
            const statusText = document.getElementById('status-text');
            const resultsContainer = document.getElementById('results-container');
            const insightsContainer = document.getElementById('insights-container');
            const papersContainer = document.getElementById('papers-container');
            const followUpQuery = document.getElementById('follow-up-query');
            const followUpButton = document.getElementById('follow-up-button');
            const followUpResults = document.getElementById('follow-up-results');
            const maxSearchInput = document.getElementById('maxSearch');
            const resultsPerQueryInput = document.getElementById('resultsPerQuery');
            const estimationContainer = document.getElementById('estimation-container');
            const estimatedTimeEl = document.getElementById('estimated-time');
            const estimatedCostEl = document.getElementById('estimated-cost');
            // Donation elements
            const donationOptions = document.querySelectorAll('.donation-option');
            const customAmountInput = document.getElementById('custom-amount');
            const donateButton = document.getElementById('donate-button');

            // Tab navigation
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => {
                tab.addEventListener('click', () => {
                    // Remove active class from all tabs
                    tabs.forEach(t => t.classList.remove('active'));
                    // Add active class to clicked tab
                    tab.classList.add('active');

                    // Hide all tab panes
                    document.querySelectorAll('.tab-pane').forEach(pane => {
                        pane.classList.remove('active');
                    });

                    // Show the corresponding tab pane
                    const tabName = tab.getAttribute('data-tab');
                    document.getElementById(`${tabName}-tab`).classList.add('active');
                });
            });



            // Get processing time and cost estimates when query changes
            queryInput.addEventListener('input', debounce(updateEstimates, 500));
            maxSearchInput.addEventListener('change', updateEstimates);
            resultsPerQueryInput.addEventListener('change', updateEstimates);

            function updateEstimates() {
                const query = queryInput.value.trim();
                if (query.length < 3) {
                    estimationContainer.classList.add('hidden');
                    return;
                }

                const maxSearch = parseInt(maxSearchInput.value);
                const resultsPerQuery = parseInt(resultsPerQueryInput.value);

                fetch('/api/TruthSeeker/estimate_processing', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query,
                        max_search: maxSearch,
                        num_search_result_per_query: resultsPerQuery
                    })
                })
                .then(res => res.json())
                .then(data => {
                    if (data.result && data.result.data) {
                        estimatedTimeEl.textContent = data.result.data.time;
                        estimatedCostEl.textContent = data.result.data.price;
                        estimationContainer.classList.remove('hidden');
                    }
                })
                .catch(err => {
                    console.error('Error getting estimates:', err);
                    estimationContainer.classList.add('hidden');
                });
            }

            // Search button click handler
            searchButton.addEventListener('click', startResearch);

            // Stop button click handler
            stopButton.addEventListener('click', stopResearch);

            // Follow-up button click handler
            followUpButton.addEventListener('click', submitFollowUp);

            // Donation option selection
            donationOptions.forEach(option => {
                option.addEventListener('click', () => {
                    if (!option.classList.contains('custom')) {
                        // Remove selected class from all options
                        donationOptions.forEach(opt => opt.classList.remove('selected'));
                        // Add selected class to clicked option
                        option.classList.add('selected');
                        // Update selected amount
                        selectedAmount = parseFloat(option.getAttribute('data-amount'));
                    }
                });
            });

            // Custom amount input handler
            customAmountInput.addEventListener('input', () => {
                // Remove selected class from all options
                donationOptions.forEach(opt => opt.classList.remove('selected'));
                // Add selected class to custom option
                customAmountInput.parentElement.classList.add('selected');
                // Update selected amount
                selectedAmount = parseFloat(customAmountInput.value) || 5;
            });

            // Donate button click handler
            donateButton.addEventListener('click', () => {
                const amount = customAmountInput.parentElement.classList.contains('selected')
                    ? parseFloat(customAmountInput.value)
                    : selectedAmount;

                if (amount < 2) {
                    alert('Minimum donation amount is €2');
                    return;
                }

                initiateStripePayment(amount);
            });

            // Functions
            function startResearch() {
                const query = queryInput.value.trim();
                if (!query) {
                    alert('Please enter a research question');
                    return;
                }



                searchButton.disabled = true;
                searchButton.innerHTML = '<div class="spinner"></div>Processing...';
                stopButton.classList.remove('hidden');
                progressContainer.classList.remove('hidden');
                statusText.classList.remove('hidden');
                statusText.textContent = 'Initializing research...';

                const maxSearch = parseInt(maxSearchInput.value);
                const resultsPerQuery = parseInt(resultsPerQueryInput.value);

                // Start the research process
                fetch('/api/TruthSeeker/start_research', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        query: query,
                        session_id: sessionId,
                        max_search: maxSearch,
                        num_search_result_per_query: resultsPerQuery
                    })
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.result && data.result.data) {
                        currentResearchId = data.result.data.research_id;
                        localStorage.setItem('currentResearchId', currentResearchId);

                        // Connect to SSE endpoint for status updates
                        connectToSSE(currentResearchId);


                    } else {
                        handleError('Failed to start research process');
                    }
                })
                .catch(error => {
                    console.error('Error starting research:', error);
                    handleError('Error starting research: ' + error.message);
                });
            }

            function connectToSSE(researchId) {
                if (eventSource) {
                    eventSource.close();
                }

                // Create EventSource connection to the SSE endpoint
                eventSource = new EventSource(`/sse/TruthSeeker/status_stream?research_id=${researchId}`);

                // Handle status update events
                eventSource.addEventListener('status_update', function(e) {
                    try {
                        const statusData = JSON.parse(e.data);
                        updateStatus(statusData);

                        // If research is complete, close the connection and load results
                        if (statusData.status === 'complete') {
                            closeSSEConnection();
                            loadResults();
                        } else if (statusData.status === 'error') {
                            closeSSEConnection();
                            handleError(statusData.info || 'An error occurred');
                        }
                    } catch (error) {
                        console.error('Error parsing SSE data:', error);
                        handleError('Error processing server update');
                    }
                });

                // Handle connection errors
                eventSource.onerror = function() {
                    console.error('SSE connection error');
                    closeSSEConnection();
                    handleError('Connection to server lost');
                };
            }

            function closeSSEConnection() {
                if (eventSource) {
                    eventSource.close();
                    eventSource = null;
                }
            }

            function stopResearch() {
                if (!currentResearchId) return;

                fetch('/api/TruthSeeker/stop_research', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        research_id: currentResearchId
                    })
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    closeSSEConnection();
                    resetUI();
                })
                .catch(error => {
                    console.error('Error stopping research:', error);
                    handleError('Error stopping research: ' + error.message);
                });
            }

            function updateStatus(statusData) {
                // Update progress bar
                if (statusData.progress !== undefined) {
                    progressBar.style.width = `${statusData.progress * 100}%`;
                }

                // Update status text
                if (statusData.step) {
                    statusText.textContent = statusData.step;
                    if (statusData.info) {
                        statusText.textContent += `: ${statusData.info}`;
                    }
                }
            }

            function loadResults() {
                if (!currentResearchId) return;

                // Reset UI
                searchButton.disabled = false;
                searchButton.textContent = 'Start Research';
                stopButton.classList.add('hidden');

                // Get research results
                fetch(`/api/TruthSeeker/research_results?research_id=${currentResearchId}`)
                    .then(response => {
                        if (!response.ok) {
                            throw new Error('Network response was not ok');
                        }
                        return response.json();
                    })
                    .then(data => {
                        if (data.result && data.result.data) {
                            displayResults(data.result.data);
                            resultsContainer.classList.remove('hidden');
                        } else {
                            handleError('No results found');
                        }
                    })
                    .catch(error => {
                        console.error('Error loading results:', error);
                        handleError('Error loading results: ' + error.message);
                    });
            }

            function submitFollowUp() {
                const query = followUpQuery.value.trim();
                if (!query || !currentResearchId) return;

                followUpButton.disabled = true;
                followUpButton.innerHTML = '<div class="spinner"></div>Processing...';

                fetch('/api/TruthSeeker/follow_up_query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        research_id: currentResearchId,
                        query: query
                    })
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    followUpButton.disabled = false;
                    followUpButton.textContent = 'Submit Question';

                    if (data.result && data.result.data) {
                        // Display follow-up results
                        const resultHtml = `
                            <div class="insight-card">
                                <h3>Follow-up: ${query}</h3>
                                <div>${formatMarkdown(data.result.data.answer)}</div>
                            </div>
                        `;
                        followUpResults.innerHTML = resultHtml + followUpResults.innerHTML;
                        followUpQuery.value = '';
                    } else {
                        alert('Failed to get follow-up answer');
                    }
                })
                .catch(error => {
                    console.error('Error with follow-up query:', error);
                    followUpButton.disabled = false;
                    followUpButton.textContent = 'Submit Question';
                    alert('Error processing follow-up query: ' + error.message);
                });
            }

            function displayResults(results) {
                // Display insights
                if (results.insights) {
                    insightsContainer.innerHTML = formatMarkdown(results.insights);
                } else {
                    insightsContainer.innerHTML = '<p>No insights generated</p>';
                }

                // Display papers
                if (results.papers && results.papers.length > 0) {
                    let papersHtml = '';
                    results.papers.forEach(paper => {
                        papersHtml += `
                            <div class="paper-item">
                                <h4>${paper.title}</h4>
                                <p>${paper.authors ? paper.authors.join(', ') : 'Unknown author(s)'}</p>
                                <p>
                                    <a href="${paper.pdf_url}" target="_blank">PDF</a> |
                                    <a href="${paper.url}" target="_blank">ArXiv</a>
                                </p>
                            </div>
                        `;
                    });
                    papersContainer.innerHTML = papersHtml;
                } else {
                    papersContainer.innerHTML = '<p>No papers found</p>';
                }
            }

            function initiateStripePayment(amount) {
                fetch('/api/TruthSeeker/create_payment', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        amount: amount,
                        session_id: sessionId
                    })
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.result && data.result.data && data.result.data.url) {
                        // Redirect to Stripe checkout
                        window.open(data.result.data.url, '_blank');

                        // Set up SSE for payment status updates
                        connectToPaymentSSE(sessionId);
                    } else {
                        alert('Failed to create payment session');
                    }
                })
                .catch(error => {
                    console.error('Error creating payment:', error);
                    alert('Error creating payment: ' + error.message);
                });
            }

            function connectToPaymentSSE(sessionId) {
                const paymentEventSource = new EventSource(`/sse/TruthSeeker/payment_stream?session_id=${sessionId}`);

                paymentEventSource.addEventListener('payment_update', function(e) {
                    try {
                        const paymentData = JSON.parse(e.data);
                        if (paymentData.status === 'completed') {
                            paymentEventSource.close();



                            alert('Payment successful! Your credit has been updated.');
                        } else if (paymentData.status === 'cancelled') {
                            paymentEventSource.close();
                        }
                    } catch (error) {
                        console.error('Error parsing payment SSE data:', error);
                    }
                });

                // Close the connection after 5 minutes if it's still open
                setTimeout(() => {
                    if (paymentEventSource.readyState === 1) { // 1 = OPEN
                        paymentEventSource.close();
                    }
                }, 300000);
            }

            function handleError(message) {
                statusText.textContent = message;
                searchButton.disabled = false;
                searchButton.textContent = 'Start Research';
                setTimeout(() => {
                    progressContainer.classList.add('hidden');
                    stopButton.classList.add('hidden');
                    statusText.classList.add('hidden');
                }, 3000);
            }

            function resetUI() {
                searchButton.disabled = false;
                searchButton.textContent = 'Start Research';
                stopButton.classList.add('hidden');
                statusText.textContent = 'Research stopped';
                progressBar.style.width = '0%';
                setTimeout(() => {
                    progressContainer.classList.add('hidden');
                    statusText.classList.add('hidden');
                }, 3000);
            }

            function generateSessionId() {
                const sessionId = 'session_' + Math.random().toString(36).substring(2, 15);
                localStorage.setItem('researchSessionId', sessionId);
                return sessionId;
            }

            function formatMarkdown(text) {
                if (!text) return '';

                // Replace headers
                text = text.replace(/^# (.*$)/gim, '<h2>$1</h2>');
                text = text.replace(/^## (.*$)/gim, '<h3>$1</h3>');
                text = text.replace(/^### (.*$)/gim, '<h4>$1</h4>');

                // Replace bold
                text = text.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');

                // Replace italic
                text = text.replace(/\*(.*?)\*/gim, '<em>$1</em>');

                // Replace lists
                text = text.replace(/^\* (.*$)/gim, '<ul><li>$1</li></ul>');
                text = text.replace(/^- (.*$)/gim, '<ul><li>$1</li></ul>');
                text = text.replace(/^(\d+)\. (.*$)/gim, '<ol><li>$2</li></ol>');

                // Replace links
                text = text.replace(/\[(.*?)\]\((.*?)\)/gim, '<a href="$2" target="_blank">$1</a>');

                // Replace paragraphs
                text = text.replace(/^\s*$/gim, '</p><p>');

                // Fix nested lists
                text = text.replace(/<\/ul>\s*<ul>/gim, '');
                text = text.replace(/<\/ol>\s*<ol>/gim, '');

                // Wrap in paragraph
                text = '<p>' + text + '</p>';

                return text;
            }

            function debounce(func, wait) {
                let timeout;
                return function() {
                    const context = this;
                    const args = arguments;
                    clearTimeout(timeout);
                    timeout = setTimeout(() => {
                        func.apply(context, args);
                    }, wait);
                };
            }

            // Check if there's a current research in progress
            const savedResearchId = localStorage.getItem('currentResearchId');
            if (savedResearchId) {
                currentResearchId = savedResearchId;
                fetch(`/api/TruthSeeker/research_status?research_id=${currentResearchId}`)
                    .then(response => {
                        if (!response.ok) {
                            throw new Error('Network response was not ok');
                        }
                        return response.json();
                    })
                    .then(data => {
                        if (data.result && data.result.data) {
                            if (data.result.data.status === 'complete') {
                                loadResults();
                            } else if (data.result.data.status === 'processing') {
                                // Research is still in progress
                                progressContainer.classList.remove('hidden');
                                statusText.classList.remove('hidden');
                                stopButton.classList.remove('hidden');
                                searchButton.disabled = true;
                                searchButton.innerHTML = '<div class="spinner"></div>Processing...';
                                connectToSSE(currentResearchId);
                            }
                        } else {
                            // Research not found, clear saved ID
                            localStorage.removeItem('currentResearchId');
                        }
                    })
                    .catch(() => {
                        // Error, clear saved ID
                        localStorage.removeItem('currentResearchId');
                    });
            }
        }, 100); // Short timeout to ensure DOM is ready
        </script>
        """

    return Result.html(app.web_context() + css + html + js)


@export(mod_name=MOD_NAME, api=True, version=version)
async def status_stream(app: App, research_id: str):
    """SSE stream endpoint for research status updates"""
    if not hasattr(app, 'sse_queues'):
        app.sse_queues = {}

    # Create a message queue for this research_id if it doesn't exist
    if research_id not in app.sse_queues:
        app.sse_queues[research_id] = asyncio.Queue()

    async def generate():
        # Send initial status
        if hasattr(app, 'research_processes') and research_id in app.research_processes:
            process = app.research_processes[research_id]
            initial_status = {
                "status": process['status'],
                "progress": process['progress'],
                "step": process['step'],
                "info": process['info']
            }
            yield f"event: status_update\ndata: {json.dumps(initial_status)}\n\n"

        try:
            # Stream status updates
            while True:
                try:
                    # Wait for a new status update with a timeout
                    status_data = await asyncio.wait_for(app.sse_queues[research_id].get(), timeout=30)
                    yield f"event: status_update\ndata: {json.dumps(status_data)}\n\n"

                    # If the research is complete or there was an error, exit the loop
                    if status_data.get('status') in ['complete', 'error', 'stopped']:
                        break
                except TimeoutError:
                    # Send a keep-alive comment to prevent connection timeout
                    yield ":\n\n"
        finally:
            # Clean up resources when the client disconnects
            if research_id in app.sse_queues:
                # Keep the queue for other potential clients
                pass

    return Result.stream(generate())


@export(mod_name=MOD_NAME, api=True, version=version)
async def payment_stream(app: App, session_id: str):
    """SSE stream endpoint for payment status updates"""
    if not hasattr(app, 'payment_queues'):
        app.payment_queues = {}

    # Create a message queue for this session_id if it doesn't exist
    if session_id not in app.payment_queues:
        app.payment_queues[session_id] = asyncio.Queue()

    async def generate():
        try:
            # Stream payment updates
            while True:
                try:
                    # Wait for a payment update with a timeout
                    payment_data = await asyncio.wait_for(app.payment_queues[session_id].get(), timeout=30)
                    yield f"event: payment_update\ndata: {json.dumps(payment_data)}\n\n"

                    # If the payment is complete or cancelled, exit the loop
                    if payment_data.get('status') in ['completed', 'cancelled']:
                        break
                except TimeoutError:
                    # Send a keep-alive comment to prevent connection timeout
                    yield ":\n\n"
        finally:
            # Clean up resources when the client disconnects
            if session_id in app.payment_queues:
                # Keep the queue for other potential clients
                pass

    return Result.stream(generate())


@export(mod_name=MOD_NAME, api=True, version=version)
async def estimate_processing(data):
    """Estimate processing time and cost for a given query"""
    # Use the static method to estimate metrics
    query, max_search, num_search_result_per_query= data.get("query", ""), data.get("max_search",4), data.get("num_search_result_per_query",6)
    estimated_time, estimated_price = ArXivPDFProcessor.estimate_processing_metrics(
        query_length=len(query),
        max_search=max_search,
        num_search_result_per_query=num_search_result_per_query,
        chunk_size=1_000_000,
        overlap=2_000,
        num_workers=None
    )

    return Result.ok(data={
        "time": estimated_time,
        "price": estimated_price
    })


@export(mod_name=MOD_NAME, api=True, version=version)
async def start_research(app: App, data):
    """Start a new research process"""
    # Get data from the request
    query = data.get("query")
    session_id = data.get("session_id")
    max_search = data.get("max_search", 4)
    num_search_result_per_query = data.get("num_search_result_per_query", 4)

    # Get the tools module
    tools = get_app("ArXivPDFProcessor").get_mod("isaa")
    if not hasattr(tools, 'initialized') or not tools.initialized:
        tools.init_isaa(build=True)

    # Generate a unique research_id
    research_id = str(uuid.uuid4())

    # Store the research information in a global dictionary
    if not hasattr(app, 'research_processes'):
        app.research_processes = {}

    # Initialize SSE queues if not already done
    if not hasattr(app, 'sse_queues'):
        app.sse_queues = {}

    # Create a queue for this research process
    app.sse_queues[research_id] = asyncio.Queue()

    # Create a processor with callback for status updates
    app.research_processes[research_id] = {
        'status': 'initializing',
        'progress': 0.0,
        'step': 'Initializing',
        'info': '',
        'query': query,
        'session_id': session_id,
        'processor': None,
        'results': None,
        'stop_requested': False
    }

    # Define the callback function that sends updates to the SSE queue
    def status_callback(status_data):
        if research_id in app.research_processes:
            process = app.research_processes[research_id]
            process['status'] = 'processing'
            process['progress'] = status_data.get('progress', 0.0)
            process['step'] = status_data.get('step', '')
            process['info'] = status_data.get('info', '')

            # Put the status update in the SSE queue
            status_update = {
                "status": process['status'],
                "progress": process['progress'],
                "step": process['step'],
                "info": process['info']
            }

            if research_id in app.sse_queues:
                asyncio.create_task(app.sse_queues[research_id].put(status_update))

    # Create the processor
    processor = ArXivPDFProcessor(
        query=query,
        tools=tools,
        chunk_size=1_000_000,
        overlap=2_000,
        max_search=max_search,
        num_search_result_per_query=num_search_result_per_query,
        download_dir=f"pdfs_{research_id}",
        callback=status_callback
    )

    app.research_processes[research_id]['processor'] = processor

    # Process in the background
    async def process_in_background():
        try:
            # Check if stop was requested before starting
            if app.research_processes[research_id]['stop_requested']:
                app.research_processes[research_id]['status'] = 'stopped'
                if research_id in app.sse_queues:
                    await app.sse_queues[research_id].put({
                        "status": "stopped",
                        "progress": 0,
                        "step": "Research stopped",
                        "info": ""
                    })
                return

            # Start processing
            papers, insights = await processor.process()

            # Check if stop was requested during processing
            if app.research_processes[research_id]['stop_requested']:
                app.research_processes[research_id]['status'] = 'stopped'
                if research_id in app.sse_queues:
                    await app.sse_queues[research_id].put({
                        "status": "stopped",
                        "progress": 1,
                        "step": "Research stopped",
                        "info": ""
                    })
                return

            # Store results
            app.research_processes[research_id]['results'] = {
                'papers': papers,
                'insights': insights['response'] if insights and 'response' in insights else None
            }
            app.research_processes[research_id]['status'] = 'complete'

            # Send final status update
            if research_id in app.sse_queues:
                await app.sse_queues[research_id].put({
                    "status": "complete",
                    "progress": 1,
                    "step": "Research complete",
                    "info": f"Found {len(papers)} papers"
                })

        except Exception as e:
            app.research_processes[research_id]['status'] = 'error'
            app.research_processes[research_id]['info'] = str(e)

            # Send error status
            if research_id in app.sse_queues:
                await app.sse_queues[research_id].put({
                    "status": "error",
                    "progress": 0,
                    "step": "Error",
                    "info": str(e)
                })

            print(f"Error in research process {research_id}: {str(e)}")

    # Start the background task
    asyncio.create_task(process_in_background())

    return Result.ok(data={"research_id": research_id})


@export(mod_name=MOD_NAME, api=True, version=version)
async def research_status(app: App, research_id: str):
    """Get the status of a research process"""
    if not hasattr(app, 'research_processes') or research_id not in app.research_processes:
        return Result.default_user_error(info="Research process not found")

    research_process = app.research_processes[research_id]

    return Result.ok(data={
        "status": research_process['status'],
        "progress": research_process['progress'],
        "step": research_process['step'],
        "info": research_process['info']
    })


@export(mod_name=MOD_NAME, api=True, version=version)
async def stop_research(app: App, data):
    """Stop a research process"""
    research_id = data.get("research_id")
    if not hasattr(app, 'research_processes') or research_id not in app.research_processes:
        return Result.default_user_error(info="Research process not found")

    app.research_processes[research_id]['stop_requested'] = True

    # Send stopped status to SSE clients
    if hasattr(app, 'sse_queues') and research_id in app.sse_queues:
        await app.sse_queues[research_id].put({
            "status": "stopped",
            "progress": app.research_processes[research_id]['progress'],
            "step": "Stopping research",
            "info": ""
        })

    return Result.ok(data={"status": "stop_requested"})


@export(mod_name=MOD_NAME, api=True, version=version)
async def research_results(app: App, research_id: str):
    """Get the results of a completed research process"""
    if not hasattr(app, 'research_processes') or research_id not in app.research_processes:
        return Result.default_user_error(info="Research process not found")

    research_process = app.research_processes[research_id]

    if research_process['status'] != 'complete':
        return Result.default_user_error(info="Research is not complete")

    return Result.ok(data=research_process['results'])


@export(mod_name=MOD_NAME, api=True, version=version)
async def follow_up_query(app: App, data):
    """Ask a follow-up question about the research"""
    research_id = data.get("research_id")
    query = data.get("query")

    if not hasattr(app, 'research_processes') or research_id not in app.research_processes:
        return Result.default_user_error(info="Research process not found")

    research_process = app.research_processes[research_id]

    if research_process['status'] != 'complete':
        return Result.default_user_error(info="Research is not complete")

    processor = research_process['processor']
    if not processor:
        return Result.default_user_error(info="Processor not available")

    try:
        # Use the extra_query method to ask follow-up questions
        result = await processor.extra_query(query)

        return Result.ok(data={"answer": result['response'] if result and 'response' in result else "No response"})
    except Exception as e:
        return Result.default_internal_error(info=f"Error processing follow-up query: {str(e)}")


@export(mod_name=MOD_NAME, api=True, version=version)
async def create_payment(app: App, data):
    """Create a Stripe payment session"""
    amount = data.get("amount")
    session_id = data.get("session_id")

    if amount < 2:
        return Result.default_user_error(info="Minimum donation amount is €2")

    try:
        # Create a Stripe Checkout Session
        base_url = f"https://{os.getenv('HOSTNAME', 'localhost:5000')}"
        success_url = f"{base_url}/api/{MOD_NAME}/payment_success?session_id={session_id}"
        cancel_url = f"{base_url}/api/{MOD_NAME}/payment_cancel?session_id={session_id}"
        stripe = __import__('stripe')
        stripe.api_key = os.getenv('STRIPE_SECRET_KEY', 'sk_test_YourSecretKey')

        stripe_session = stripe.checkout.Session.create(
            payment_method_types=['card', 'link'],
            line_items=[{
                'price_data': {
                    'currency': 'eur',
                    'product_data': {'name': 'Research Credits'},
                    'unit_amount': int(amount * 100),
                },
                'quantity': 1,
            }],
            automatic_tax={"enabled": True},
            mode='payment',
            success_url=success_url,
            cancel_url=cancel_url
        )

        # Store the payment info
        if not hasattr(app, 'payment_info'):
            app.payment_info = {}

        # Initialize payment_queues if not already done
        if not hasattr(app, 'payment_queues'):
            app.payment_queues = {}

        # Create a queue for this payment
        app.payment_queues[session_id] = asyncio.Queue()

        app.payment_info[session_id] = {
            'payment_id': stripe_session.id,
            'amount': amount,
            'status': 'pending'
        }

        return Result.ok(data={"url": stripe_session.url})
    except Exception as e:
        return Result.default_internal_error(info=f"Error creating payment: {str(e)}")


@export(mod_name=MOD_NAME, api=True, version=version)
async def payment_success(app: App, session_id: str, request_as_kwarg=True, request=None):
    """Handle successful payment"""
    if not hasattr(app, 'payment_info') or session_id not in app.payment_info:
        return Result.html(app.web_context() + """
        <div style="text-align: center; padding: 50px;">
            <h2>Payment Session Not Found</h2>
            <p>Return to the main page to continue.</p>
            <a href="/" style="display: inline-block; margin-top: 20px; padding: 10px 20px; background-color: #4F46E5; color: white; text-decoration: none; border-radius: 5px;">Return to Home</a>
        </div>
        """)

    payment_info = app.payment_info[session_id]

    try:
        # Verify the payment with Stripe
        stripe = __import__('stripe')
        stripe.api_key = os.getenv('STRIPE_SECRET_KEY', 'sk_test_YourSecretKey')

        stripe_session = stripe.checkout.Session.retrieve(payment_info['payment_id'])

        if stripe_session.payment_status == 'paid':
            payment_info['status'] = 'completed'

            # Notify SSE clients about payment completion
            if hasattr(app, 'payment_queues') and session_id in app.payment_queues:
                await app.payment_queues[session_id].put({
                    "status": "completed",
                    "amount": payment_info['amount']
                })

            return Result.html(app.web_context() + """
            <div style="text-align: center; padding: 50px;">
                <h2>Thank You for Your Support!</h2>
                <p>Your payment was successful. You can now close this window and continue with your research.</p>
                <script>
                    setTimeout(function() {
                        window.close();
                    }, 5000);
                </script>
            </div>
            """)
        else:
            return Result.html(app.web_context() + """
            <div style="text-align: center; padding: 50px;">
                <h2>Payment Not Completed</h2>
                <p>Your payment has not been completed. Please try again.</p>
                <button onclick="window.close()">Close Window</button>
            </div>
            """)
    except Exception as e:
        return Result.html(app.web_context() + f"""
        <div style="text-align: center; padding: 50px;">
            <h2>Error Processing Payment</h2>
            <p>There was an error processing your payment: {str(e)}</p>
            <button onclick="window.close()">Close Window</button>
        </div>
        """)


@export(mod_name=MOD_NAME, api=True, version=version)
async def payment_cancel(app: App, session_id: str, request_as_kwarg=True, request=None):
    """Handle cancelled payment"""
    if hasattr(app, 'payment_info') and session_id in app.payment_info:
        app.payment_info[session_id]['status'] = 'cancelled'

        # Notify SSE clients about payment cancellation
        if hasattr(app, 'payment_queues') and session_id in app.payment_queues:
            await app.payment_queues[session_id].put({
                "status": "cancelled"
            })

    return Result.html(app.web_context() + """
    <div style="text-align: center; padding: 50px;">
        <h2>Payment Cancelled</h2>
        <p>Your payment was cancelled.</p>
        <script>
            setTimeout(function() {
                window.close();
            }, 3000);
        </script>
    </div>
    """)


@export(mod_name=MOD_NAME, version=version, exit_f=True)
def cleanup_module(app: App):
    """Cleanup resources when the module is unloaded"""
    # Clean up any temp files or resources
    import glob
    import shutil

    # Remove temporary PDF directories
    for pdf_dir in glob.glob("pdfs_*"):
        try:
            shutil.rmtree(pdf_dir)
        except Exception as e:
            print(f"Error removing directory {pdf_dir}: {str(e)}")

    # Clear any SSE queues
    if hasattr(app, 'sse_queues'):
        app.sse_queues = {}

    if hasattr(app, 'payment_queues'):
        app.payment_queues = {}

    return Result.ok(info="ArXivPDFProcessor UI cleaned up")
