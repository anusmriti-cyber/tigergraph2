/* ============================================================
   Dashboard Logic – Query, Responses, Metrics, Charts
   ============================================================ */

let tokenChartInst, accuracyChartInst, latencyChartInst, costChartInst, radarChartInst;

// --- Mock pipeline data ---
const MOCK_DATA = {
    default: {
        llm: {
            response: "CRISPR-Cas9 is a genome editing tool adapted from bacterial immune systems. The Cas9 protein creates double-strand breaks at specific DNA locations guided by a synthetic RNA. In oncology, this enables precise knockout of oncogenes, correction of tumor-suppressor mutations, and engineering of CAR-T cells for immunotherapy. Clinical trials are exploring ex-vivo editing of patient T-cells to enhance anti-tumor responses.",
            tokens: 1850, latency: 1240, cost: 0.0370, bert_score: 62, llm_judge: "FAIL"
        },
        rag: {
            response: "CRISPR-Cas9 gene editing in oncology operates through a ribonucleoprotein complex where the Cas9 endonuclease is directed by a single guide RNA (sgRNA) to create site-specific double-strand breaks (DSBs). In cancer research, this mechanism is leveraged for: (1) functional genomic screens to identify essential cancer genes, (2) precise knockout of oncogenes like KRAS and MYC, (3) restoration of tumor suppressor function in TP53-mutant cells, and (4) engineering enhanced CAR-T cells with PD-1 knockout for improved immunotherapy efficacy. Retrieved from 12 relevant papers.",
            tokens: 1420, latency: 1680, cost: 0.0284, bert_score: 81, llm_judge: "PASS"
        },
        graph: {
            response: "CRISPR-Cas9 in oncology functions via a multi-pathway mechanism: The Cas9-sgRNA complex induces DSBs at target loci, repaired via NHEJ (gene knockout) or HDR (precise correction). Knowledge graph analysis reveals key relationships: CRISPR → targets KRAS (pancreatic adenocarcinoma, 34 papers), TP53 restoration (breast cancer, 28 papers), PD-1/PD-L1 knockout in CAR-T engineering (melanoma/lymphoma, 41 papers). Graph traversal identifies emerging connections: CRISPR combined with base editing shows 73% improved specificity over wild-type Cas9. Cross-entity analysis links CRISPR efficacy to delivery vectors (lipid nanoparticles > viral vectors for solid tumors). Sourced from 47 interconnected entities across 89 papers.",
            tokens: 980, latency: 2150, cost: 0.0196, bert_score: 94, llm_judge: "PASS"
        }
    }
};

function useExample(el) {
    document.getElementById('queryInput').value = el.textContent;
    document.getElementById('queryInput').focus();
}

function submitQuery() {
    const input = document.getElementById('queryInput');
    if (!input.value.trim()) { input.focus(); return; }
    document.getElementById('queryBtn').disabled = true;
    document.getElementById('loadingOverlay').classList.remove('hidden');
    ['responsesSection','metricsSummary','chartsSection','verdictBanner','summarySection'].forEach(id =>
        document.getElementById(id).classList.add('hidden'));
        
    fetch('http://localhost:8000/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: input.value.trim() })
    })
    .then(response => {
        if (!response.ok) throw new Error("API Error");
        return response.json();
    })
    .then(data => {
        document.getElementById('loadingOverlay').classList.add('hidden');
        document.getElementById('queryBtn').disabled = false;
        renderResults(data);
    })
    .catch(error => {
        console.error(error);
        alert("Error connecting to Backend API. Please ensure the backend server is running.");
        document.getElementById('loadingOverlay').classList.add('hidden');
        document.getElementById('queryBtn').disabled = false;
    });
}

document.getElementById('queryInput')?.addEventListener('keydown', e => {
    if (e.key === 'Enter') submitQuery();
});

// --- Typewriter effect ---
function typeWriter(el, text, speed = 12) {
    el.innerHTML = '';
    let i = 0;
    const cursor = document.createElement('span');
    cursor.className = 'type-cursor';
    cursor.textContent = '▋';
    cursor.style.cssText = 'animation:blink 0.7s step-end infinite;color:var(--cyan);font-weight:400;';
    el.appendChild(cursor);
    const style = document.createElement('style');
    style.textContent = '@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}';
    if (!document.querySelector('style[data-cursor]')) { style.dataset.cursor = '1'; document.head.appendChild(style); }

    function tick() {
        if (i < text.length) {
            cursor.before(text.charAt(i));
            i++;
            setTimeout(tick, speed);
        } else {
            cursor.remove();
        }
    }
    tick();
}

// --- Count-up animation ---
function animateCount(el, target, suffix = '', prefix = '', duration = 1000) {
    const start = performance.now();
    const isFloat = String(target).includes('.');
    function frame(now) {
        const progress = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = isFloat ? (target * eased).toFixed(4) : Math.round(target * eased);
        el.textContent = prefix + (isFloat ? current : Number(current).toLocaleString()) + suffix;
        if (progress < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}

// --- Render results ---
function renderResults(data) {
    // Typewriter responses
    typeWriter(document.getElementById('responseLLM'), data.llm.response, 8);
    setTimeout(() => typeWriter(document.getElementById('responseRAG'), data.rag.response, 8), 300);
    setTimeout(() => typeWriter(document.getElementById('responseGraph'), data.graph.response, 8), 600);

    // Animated metrics
    setTimeout(() => {
        animateMetrics('LLM', data.llm);
        animateMetrics('RAG', data.rag);
        animateMetrics('Graph', data.graph);
        
        // Static metrics
        document.getElementById('llmJudgeLLM').textContent = data.llm.llm_judge;
        document.getElementById('llmJudgeRAG').textContent = data.rag.llm_judge;
        document.getElementById('llmJudgeGraph').textContent = data.graph.llm_judge;
    }, 400);

    // Accuracy bars
    setTimeout(() => {
        document.getElementById('accuracyBarLLM').style.setProperty('--bar-width', data.llm.bert_score + '%');
        document.getElementById('accuracyBarRAG').style.setProperty('--bar-width', data.rag.bert_score + '%');
        document.getElementById('accuracyBarGraph').style.setProperty('--bar-width', data.graph.bert_score + '%');
    }, 600);

    // Winner detection
    detectWinner(data);

    // Show sections staggered
    document.getElementById('responsesSection').classList.remove('hidden');
    setTimeout(() => document.getElementById('verdictBanner').classList.remove('hidden'), 300);
    setTimeout(() => document.getElementById('metricsSummary').classList.remove('hidden'), 500);
    setTimeout(() => {
        document.getElementById('chartsSection').classList.remove('hidden');
        renderCharts(data);
    }, 700);
    // Show AI Summary
    setTimeout(() => {
        document.getElementById('summarySection').classList.remove('hidden');
        if (data.summary) {
            typeWriter(document.getElementById('summaryBody'), data.summary, 10);
        } else {
            document.getElementById('summaryBody').textContent = 'Summary not available.';
        }
    }, 900);

    document.getElementById('responsesSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function animateMetrics(suffix, d) {
    animateCount(document.getElementById('tokens' + suffix), d.tokens, '', '', 1200);
    animateCount(document.getElementById('latency' + suffix), d.latency, 'ms', '', 1200);
    animateCount(document.getElementById('cost' + suffix), d.cost, '', '$', 1200);
    animateCount(document.getElementById('accuracy' + suffix), d.bert_score, '', '', 1200);
}

function detectWinner(data) {
    const pipes = [
        { key: 'LLM', name: 'LLM-Only', ...data.llm },
        { key: 'RAG', name: 'Standard RAG', ...data.rag },
        { key: 'Graph', name: 'GraphRAG', ...data.graph }
    ];
    const best = pipes.reduce((a, b) => a.bert_score > b.bert_score ? a : b);
    const fastest = pipes.reduce((a, b) => a.latency < b.latency ? a : b);
    const cheapest = pipes.reduce((a, b) => a.cost < b.cost ? a : b);
    const reduction = Math.round((1 - data.graph.tokens / data.llm.tokens) * 100);

    // Winner badges
    ['LLM','RAG','Graph'].forEach(k => {
        document.getElementById('winner' + k).classList.add('hidden');
        document.getElementById('card' + k).classList.remove('is-winner');
    });
    document.getElementById('winner' + best.key).classList.remove('hidden');
    document.getElementById('card' + best.key).classList.add('is-winner');

    // Verdict banner
    document.getElementById('verdictTitle').textContent = best.name + ' Wins!';
    document.getElementById('verdictDesc').textContent =
        `Highest BERTScore with ${reduction}% fewer tokens than baseline`;
    document.getElementById('verdictScore').textContent = best.bert_score;

    // Summary stats
    document.getElementById('bestAccuracyValue').textContent = best.name + ' (' + best.bert_score + ')';
    document.getElementById('fastestValue').textContent = fastest.name + ' (' + fastest.latency + 'ms)';
    document.getElementById('cheapestValue').textContent = cheapest.name + ' ($' + cheapest.cost.toFixed(4) + ')';
    document.getElementById('tokenSavedValue').textContent = reduction + '% fewer tokens';
}

// --- Charts ---
function chartColors() {
    return {
        llm:   { bg: 'rgba(59,130,246,0.7)', border: '#3b82f6', bgLight: 'rgba(59,130,246,0.15)' },
        rag:   { bg: 'rgba(0,240,255,0.7)',   border: '#00f0ff', bgLight: 'rgba(0,240,255,0.15)' },
        graph: { bg: 'rgba(168,85,247,0.7)',   border: '#a855f7', bgLight: 'rgba(168,85,247,0.15)' }
    };
}

function baseOpts(yLabel) {
    return {
        responsive: true, maintainAspectRatio: false,
        animation: { duration: 1200, easing: 'easeOutQuart' },
        plugins: {
            legend: { labels: { color: 'rgba(240,240,245,0.6)', font: { family: "'Space Grotesk'", size: 11 }, boxWidth: 12, padding: 14 } },
            tooltip: { backgroundColor: 'rgba(6,6,14,0.92)', titleColor: '#f0f0f5', bodyColor: 'rgba(240,240,245,0.8)', borderColor: 'rgba(255,255,255,0.08)', borderWidth: 1, cornerRadius: 8, padding: 10 }
        },
        scales: {
            x: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: 'rgba(240,240,245,0.45)', font: { size: 11 } } },
            y: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: 'rgba(240,240,245,0.45)', font: { size: 11 } }, title: { display: !!yLabel, text: yLabel, color: 'rgba(240,240,245,0.35)', font: { size: 11 } } }
        }
    };
}

function ds(label, value, c) {
    return { label, data: [value], backgroundColor: c.bg, borderColor: c.border, borderWidth: 1.5, borderRadius: 8, barPercentage: 0.55 };
}

function renderCharts(data) {
    const c = chartColors();
    const labels = ['Pipeline Comparison'];
    [tokenChartInst, accuracyChartInst, latencyChartInst, costChartInst, radarChartInst].forEach(ch => ch?.destroy());

    // --- Radar Chart ---
    const maxTokens = Math.max(data.llm.tokens, data.rag.tokens, data.graph.tokens);
    const maxLatency = Math.max(data.llm.latency, data.rag.latency, data.graph.latency);
    const maxCost = Math.max(data.llm.cost, data.rag.cost, data.graph.cost);

    const normalize = (v, max, invert) => invert ? Math.round((1 - v / max) * 100) : Math.round(v / max * 100);

    radarChartInst = new Chart(document.getElementById('radarChart'), {
        type: 'radar',
        data: {
            labels: ['BERTScore', 'Token Efficiency', 'Speed', 'Cost Efficiency', 'Overall'],
            datasets: [
                { label: 'LLM-Only', data: [data.llm.bert_score, normalize(data.llm.tokens, maxTokens, true), normalize(data.llm.latency, maxLatency, true), normalize(data.llm.cost, maxCost, true), data.llm.bert_score * 0.8], backgroundColor: c.llm.bgLight, borderColor: c.llm.border, borderWidth: 2, pointBackgroundColor: c.llm.border, pointRadius: 4 },
                { label: 'Standard RAG', data: [data.rag.bert_score, normalize(data.rag.tokens, maxTokens, true), normalize(data.rag.latency, maxLatency, true), normalize(data.rag.cost, maxCost, true), data.rag.bert_score * 0.9], backgroundColor: c.rag.bgLight, borderColor: c.rag.border, borderWidth: 2, pointBackgroundColor: c.rag.border, pointRadius: 4 },
                { label: 'GraphRAG', data: [data.graph.bert_score, normalize(data.graph.tokens, maxTokens, true), normalize(data.graph.latency, maxLatency, true), normalize(data.graph.cost, maxCost, true), data.graph.bert_score], backgroundColor: c.graph.bgLight, borderColor: c.graph.border, borderWidth: 2, pointBackgroundColor: c.graph.border, pointRadius: 4 }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            animation: { duration: 1400, easing: 'easeOutQuart' },
            scales: {
                r: {
                    beginAtZero: true, max: 100,
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    angleLines: { color: 'rgba(255,255,255,0.06)' },
                    pointLabels: { color: 'rgba(240,240,245,0.6)', font: { family: "'Space Grotesk'", size: 12 } },
                    ticks: { display: false, stepSize: 25 }
                }
            },
            plugins: {
                legend: { labels: { color: 'rgba(240,240,245,0.6)', font: { family: "'Space Grotesk'", size: 11 }, boxWidth: 12, padding: 14 } },
                tooltip: { backgroundColor: 'rgba(6,6,14,0.92)', titleColor: '#f0f0f5', bodyColor: 'rgba(240,240,245,0.8)', borderColor: 'rgba(255,255,255,0.08)', borderWidth: 1, cornerRadius: 8 }
            }
        }
    });

    // Bar charts
    tokenChartInst = new Chart(document.getElementById('tokenChart'), {
        type: 'bar', data: { labels, datasets: [ds('LLM-Only', data.llm.tokens, c.llm), ds('Standard RAG', data.rag.tokens, c.rag), ds('GraphRAG', data.graph.tokens, c.graph)] }, options: baseOpts('Tokens Used')
    });

    const accOpts = baseOpts('BERTScore');
    accOpts.scales.y.max = 100;
    accuracyChartInst = new Chart(document.getElementById('accuracyChart'), {
        type: 'bar', data: { labels, datasets: [ds('LLM-Only', data.llm.bert_score, c.llm), ds('Standard RAG', data.rag.bert_score, c.rag), ds('GraphRAG', data.graph.bert_score, c.graph)] }, options: accOpts
    });

    latencyChartInst = new Chart(document.getElementById('latencyChart'), {
        type: 'bar', data: { labels, datasets: [ds('LLM-Only', data.llm.latency, c.llm), ds('Standard RAG', data.rag.latency, c.rag), ds('GraphRAG', data.graph.latency, c.graph)] }, options: baseOpts('Latency (ms)')
    });

    costChartInst = new Chart(document.getElementById('costChart'), {
        type: 'bar', data: { labels, datasets: [ds('LLM-Only', data.llm.cost, c.llm), ds('Standard RAG', data.rag.cost, c.rag), ds('GraphRAG', data.graph.cost, c.graph)] }, options: baseOpts('Cost (USD)')
    });
}
