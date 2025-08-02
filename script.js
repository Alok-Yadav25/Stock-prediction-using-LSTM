// Global variables
let isAnalyzing = false;
let currentStock = '';

// DOM elements
const predictionForm = document.getElementById('predictionForm');
const stockInput = document.getElementById('stockId');
const analyzeBtn = document.querySelector('.analyze-btn');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const loadingText = document.getElementById('loadingText');
const backToTopBtn = document.getElementById('backToTop');

// Loading messages
const loadingMessages = [
    "Initializing AI models...",
    "Fetching market data...",
    "Processing historical prices...",
    "Running predictions...",
    "Generating visualizations...",
    "Finalizing analysis..."
];

// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Form submission
    predictionForm.addEventListener('submit', handleFormSubmission);
    
    // Back to top functionality
    window.addEventListener('scroll', handleScroll);
    backToTopBtn.addEventListener('click', scrollToTop);
    
    // Smooth scrolling for navigation
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', handleNavClick);
    });
    
    // Initialize animations
    animateOnScroll();
}

async function handleFormSubmission(e) {
    e.preventDefault();
    
    if (isAnalyzing) return;
    
    const stockSymbol = stockInput.value.trim().toUpperCase();
    if (!stockSymbol) {
        showError('Please enter a valid stock symbol');
        return;
    }
    
    currentStock = stockSymbol;
    startAnalysis();
    
    try {
        const response = await fetch('/complete_analysis', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ stock_id: stockSymbol })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayResults(data);
        } else {
            showError(data.error || 'Analysis failed');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        stopAnalysis();
    }
}

function startAnalysis() {
    isAnalyzing = true;
    analyzeBtn.classList.add('loading');
    
    // Hide previous results
    hideAllSections();
    
    // Show loading section
    loadingSection.classList.remove('hidden');
    
    // Animate loading messages
    animateLoadingMessages();
    
    // Scroll to loading section
    setTimeout(() => {
        loadingSection.scrollIntoView({ behavior: 'smooth' });
    }, 300);
}

function stopAnalysis() {
    isAnalyzing = false;
    analyzeBtn.classList.remove('loading');
    loadingSection.classList.add('hidden');
}

function animateLoadingMessages() {
    let messageIndex = 0;
    
    const interval = setInterval(() => {
        if (!isAnalyzing) {
            clearInterval(interval);
            return;
        }
        
        loadingText.style.opacity = '0';
        
        setTimeout(() => {
            loadingText.textContent = loadingMessages[messageIndex];
            loadingText.style.opacity = '1';
            messageIndex = (messageIndex + 1) % loadingMessages.length;
        }, 300);
    }, 2000);
}

function displayResults(data) {
    hideAllSections();
    
    // Create metrics cards
    createMetricsCards(data);
    
    // Display charts
    displayCharts(data);
    
    // Display tables
    displayTables(data);
    
    // Show results section
    resultsSection.classList.remove('hidden');
    
    // Animate elements
    animateResults();
    
    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }, 300);
}

function createMetricsCards(data) {
    const metricsGrid = document.getElementById('metricsGrid');
    
    const metrics = [
        {
            title: 'Current Price',
            value: `$${data.current_price.toFixed(2)}`,
            icon: 'fas fa-dollar-sign',
            color: '#4299e1',
            change: null
        },
        {
            title: 'Predicted Price',
            value: `$${data.predicted_price.toFixed(2)}`,
            icon: 'fas fa-crystal-ball',
            color: '#9f7aea',
            change: null
        },
        {
            title: 'Price Change',
            value: `${data.price_change.toFixed(2)}%`,
            icon: data.price_change >= 0 ? 'fas fa-trending-up' : 'fas fa-trending-down',
            color: data.price_change >= 0 ? '#48bb78' : '#f56565',
            change: data.price_change >= 0 ? 'positive' : 'negative'
        },
        {
            title: 'Model RMSE',
            value: data.rmse.toFixed(2),
            icon: 'fas fa-chart-line',
            color: '#ed8936',
            change: null
        }
    ];
    
    metricsGrid.innerHTML = metrics.map(metric => `
        <div class="metric-card" style="--metric-color: ${metric.color}; --metric-color-light: ${metric.color}40">
            <div class="metric-header">
                <span class="metric-title">${metric.title}</span>
                <i class="${metric.icon} metric-icon"></i>
            </div>
            <div class="metric-value">${metric.value}</div>
            ${metric.change ? `
                <div class="metric-change ${metric.change}">
                    <i class="fas fa-${metric.change === 'positive' ? 'arrow-up' : 'arrow-down'}"></i>
                    ${metric.change === 'positive' ? 'Bullish' : 'Bearish'} Prediction
                </div>
            ` : ''}
        </div>
    `).join('');
}

function displayCharts(data) {
    // Comparison chart
    const comparisonChart = document.getElementById('comparisonChart');
    const comparisonContent = comparisonChart.querySelector('.chart-content');
    comparisonContent.innerHTML = `
        <img src="data:image/png;base64,${data.comparison_graph}" 
             alt="Original vs Predicted Prices" 
             class="chart-image">
    `;
    
    // Moving average chart
    const maChart = document.getElementById('movingAverageChart');
    const maContent = maChart.querySelector('.chart-content');
    maContent.innerHTML = `
        <img src="data:image/png;base64,${data.ma_graph}" 
             alt="Moving Averages Chart" 
             class="chart-image">
    `;
}

function displayTables(data) {
    // Price analysis table
    const priceTable = document.getElementById('priceAnalysisTable');
    const priceContent = priceTable.querySelector('.table-content');
    priceContent.innerHTML = data.price_table;
    
    // Prediction comparison table
    const predictionTable = document.getElementById('predictionComparisonTable');
    const predictionContent = predictionTable.querySelector('.table-content');
    predictionContent.innerHTML = data.prediction_table;
}

function animateResults() {
    // Animate metric cards
    const metricCards = document.querySelectorAll('.metric-card');
    metricCards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
        card.classList.add('fade-in');
    });
    
    // Animate chart cards
    const chartCards = document.querySelectorAll('.chart-card');
    chartCards.forEach((card, index) => {
        card.style.animationDelay = `${(metricCards.length + index) * 0.1}s`;
        card.classList.add('fade-in');
    });
    
    // Animate table cards
    const tableCards = document.querySelectorAll('.table-card');
    tableCards.forEach((card, index) => {
        card.style.animationDelay = `${(metricCards.length + chartCards.length + index) * 0.1}s`;
        card.classList.add('fade-in');
    });
}

function showError(message) {
    hideAllSections();
    
    const errorMessage = document.getElementById('errorMessage');
    errorMessage.textContent = message;
    
    errorSection.classList.remove('hidden');
    errorSection.scrollIntoView({ behavior: 'smooth' });
}

function hideError() {
    errorSection.classList.add('hidden');
}

function hideAllSections() {
    loadingSection.classList.add('hidden');
    resultsSection.classList.add('hidden');
    errorSection.classList.add('hidden');
}

function handleScroll() {
    // Back to top button
    if (window.pageYOffset > 300) {
        backToTopBtn.classList.add('visible');
    } else {
        backToTopBtn.classList.remove('visible');
    }
    
    // Navbar scroll effect
    const navbar = document.querySelector('.navbar');
    if (window.pageYOffset > 100) {
        navbar.style.background = 'rgba(255, 255, 255, 0.98)';
        navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.1)';
    } else {
        navbar.style.background = 'rgba(255, 255, 255, 0.95)';
        navbar.style.boxShadow = 'none';
    }
}

function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

function handleNavClick(e) {
    e.preventDefault();
    const targetId = e.target.getAttribute('href');
    if (targetId && targetId.startsWith('#')) {
        const targetElement = document.querySelector(targetId);
        if (targetElement) {
            targetElement.scrollIntoView({ behavior: 'smooth' });
        }
    }
}

function animateOnScroll() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });
    
    // Observe elements for animation
    document.querySelectorAll('.search-card, .metric-card, .chart-card, .table-card').forEach(el => {
        observer.observe(el);
    });
}

function exportTable(tableType) {
    // Simple CSV export functionality
    const table = document.querySelector(`#${tableType}Table .data-table`);
    if (!table) return;
    
    let csv = '';
    const rows = table.querySelectorAll('tr');
    
    rows.forEach(row => {
        const cells = row.querySelectorAll('th, td');
        const rowData = Array.from(cells).map(cell => {
            return '"' + cell.textContent.replace(/"/g, '""') + '"';
        }).join(',');
        csv += rowData + '\n';
    });
    
    // Download CSV
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${currentStock}_${tableType}_${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

// Utility functions
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(value);
}

function formatPercentage(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 2
    }).format(value / 100);
}

// Error handling
window.addEventListener('error', function(e) {
    console.error('JavaScript error:', e.error);
    if (isAnalyzing) {
        showError('An unexpected error occurred. Please try again.');
        stopAnalysis();
    }
});

// Prevent form submission on Enter if already analyzing
document.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && isAnalyzing) {
        e.preventDefault();
    }
});
