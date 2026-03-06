// server.js - OpenAI to NVIDIA NIM API Proxy
// Updated: March 2026 — with latest NIM models

const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// ─── Middleware ───────────────────────────────────────────────────────────────
app.use(cors());
app.use(express.json({ limit: '50mb' })); // Increased limit for multimodal (base64 images)

// ─── NVIDIA NIM Configuration ────────────────────────────────────────────────
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY  = process.env.NIM_API_KEY;

if (!NIM_API_KEY) {
  console.warn('⚠️  WARNING: NIM_API_KEY environment variable is not set.');
}

// ─── Feature Toggles ─────────────────────────────────────────────────────────
// Shows/hides <think>…</think> reasoning blocks in the response
const SHOW_REASONING = process.env.SHOW_REASONING !== 'false'; // default: true

// Enables chat_template_kwargs thinking param for models that support it
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true'; // default: false

// ─── Model Catalog ───────────────────────────────────────────────────────────
// Models are grouped by provider for easy maintenance.
// Each entry: { id, label, multimodal?, thinking? }
const NIM_MODELS = {
  // ── DeepSeek ──────────────────────────────────────────────────────────────
  'deepseek-ai/deepseek-v3.2': {
    label: 'DeepSeek V3.2 (685B · Sparse MoE · Agentic)',
    thinking: false,
  },
  'deepseek-ai/deepseek-r1': {
    label: 'DeepSeek R1 (671B · Reasoning)',
    thinking: true,
  },
  'deepseek-ai/deepseek-r1-distill-qwen-32b': {
    label: 'DeepSeek R1 Distill Qwen 32B',
    thinking: true,
  },
  'deepseek-ai/deepseek-r1-distill-qwen-14b': {
    label: 'DeepSeek R1 Distill Qwen 14B',
    thinking: true,
  },
  'deepseek-ai/deepseek-r1-distill-llama-8b': {
    label: 'DeepSeek R1 Distill Llama 8B',
    thinking: true,
  },

  // ── Qwen ──────────────────────────────────────────────────────────────────
  'qwen/qwen3.5-397b-a17b': {
    label: 'Qwen 3.5 VLM (397B MoE · Vision + Chat + Agentic)',
    multimodal: true,
    thinking: false,
  },
  'qwen/qwen3-coder-480b-a35b-instruct': {
    label: 'Qwen3 Coder 480B (Code · Instruct)',
    thinking: false,
  },
  'qwen/qwen3-next-80b-a3b-thinking': {
    label: 'Qwen3 Next 80B Thinking (MoE · Hybrid Reasoning)',
    thinking: true,
  },
  'qwen/qwq-32b': {
    label: 'QwQ 32B (Reasoning)',
    thinking: true,
  },

  // ── Kimi / MoonshotAI ─────────────────────────────────────────────────────
  'moonshotai/kimi-k2.5': {
    label: 'Kimi K2.5 (1T · Multimodal MoE · Vision + Video)',
    multimodal: true,
    thinking: true,
  },
  'moonshotai/kimi-k2-instruct': {
    label: 'Kimi K2 Instruct (128K · Text)',
    thinking: false,
  },
  'moonshotai/kimi-k2-thinking': {
    label: 'Kimi K2 Thinking (256K · Reasoning · INT4)',
    thinking: true,
  },

  // ── NVIDIA / Meta Llama ───────────────────────────────────────────────────
  'nvidia/llama-3.1-nemotron-ultra-253b-v1': {
    label: 'Llama 3.1 Nemotron Ultra 253B',
    thinking: false,
  },
  'meta/llama-4-scout-17b-16e-instruct': {
    label: 'Llama 4 Scout 17B (16 Experts)',
    multimodal: true,
    thinking: false,
  },
  'meta/llama-4-maverick-17b-128e-instruct': {
    label: 'Llama 4 Maverick 17B (128 Experts)',
    multimodal: true,
    thinking: false,
  },

  // ── OpenAI via NIM ────────────────────────────────────────────────────────
  'openai/gpt-oss-120b': {
    label: 'OpenAI GPT OSS 120B',
    thinking: false,
  },
  'openai/gpt-oss-20b': {
    label: 'OpenAI GPT OSS 20B',
    thinking: false,
  },
};

// ─── Legacy model aliases (OpenAI-style names → NIM model IDs) ───────────────
// Useful if you're migrating existing clients that use OpenAI model names.
const MODEL_MAPPING = {
  'gpt-3.5-turbo':   'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4':           'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo':     'moonshotai/kimi-k2.5',
  'gpt-4o':          'deepseek-ai/deepseek-v3.2',
  'o1':              'deepseek-ai/deepseek-r1',
  'o1-mini':         'deepseek-ai/deepseek-r1-distill-qwen-32b',
  'claude-3-opus':   'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro':      'qwen/qwen3-next-80b-a3b-thinking',
  'gemini-1.5-pro':  'qwen/qwen3.5-397b-a17b',
  'gemini-2.0':      'moonshotai/kimi-k2-thinking',
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

/**
 * Resolves a requested model name to the actual NIM model ID.
 * Priority: exact NIM ID → legacy alias → original value (fallback)
 */
function resolveModel(requestedModel) {
  if (!requestedModel) return Object.keys(NIM_MODELS)[0];
  if (NIM_MODELS[requestedModel]) return requestedModel;
  return MODEL_MAPPING[requestedModel] || requestedModel;
}

/**
 * Returns model metadata or an empty object if unknown.
 */
function getModelMeta(nimModelId) {
  return NIM_MODELS[nimModelId] || {};
}

/**
 * Strips <think>…</think> reasoning blocks from a text string
 * when SHOW_REASONING is false.
 */
function maybeStripReasoning(text) {
  if (SHOW_REASONING || !text) return text;
  return text.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
}

/**
 * Builds the extra body params for models that support thinking mode.
 */
function buildThinkingParams(meta) {
  if (!ENABLE_THINKING_MODE || !meta.thinking) return {};
  return {
    chat_template_kwargs: { thinking: true },
  };
}

// ─── Routes ──────────────────────────────────────────────────────────────────

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    nim_api_base: NIM_API_BASE,
    api_key_configured: !!NIM_API_KEY,
    show_reasoning: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    model_count: Object.keys(NIM_MODELS).length,
  });
});

// List all available NIM models (OpenAI-compatible format)
app.get('/v1/models', (req, res) => {
  const now = Math.floor(Date.now() / 1000);
  const models = Object.entries(NIM_MODELS).map(([id, meta]) => ({
    id,
    object: 'model',
    created: now,
    owned_by: id.split('/')[0],
    description: meta.label,
    capabilities: {
      multimodal: !!meta.multimodal,
      thinking: !!meta.thinking,
    },
  }));

  // Also expose legacy aliases as virtual entries
  const aliases = Object.entries(MODEL_MAPPING).map(([alias, target]) => ({
    id: alias,
    object: 'model',
    created: now,
    owned_by: 'alias',
    description: `Alias → ${target} (${NIM_MODELS[target]?.label || target})`,
    capabilities: NIM_MODELS[target]
      ? { multimodal: !!NIM_MODELS[target].multimodal, thinking: !!NIM_MODELS[target].thinking }
      : {},
  }));

  res.json({ object: 'list', data: [...models, ...aliases] });
});

// Chat completions (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  if (!NIM_API_KEY) {
    return res.status(500).json({
      error: { message: 'NIM_API_KEY is not configured on the server.', type: 'configuration_error' },
    });
  }

  try {
    const { model: requestedModel, stream = false, ...rest } = req.body;

    const nimModel = resolveModel(requestedModel);
    const meta     = getModelMeta(nimModel);

    const payload = {
      model: nimModel,
      stream,
      ...buildThinkingParams(meta),
      ...rest,
    };

    const nimResponse = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      payload,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json',
          Accept: stream ? 'text/event-stream' : 'application/json',
        },
        responseType: stream ? 'stream' : 'json',
        timeout: 120_000, // 2 min — reasoning models can be slow
      }
    );

    // ── Streaming response ─────────────────────────────────────────────────
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-NIM-Model', nimModel);

      nimResponse.data.on('data', (chunk) => {
        const raw = chunk.toString();

        if (!SHOW_REASONING) {
          // Strip <think> blocks from SSE delta content on the fly
          const cleaned = raw.replace(
            /"content"\s*:\s*"((?:[^"\\]|\\.)*)"/g,
            (match, content) => {
              const decoded = content.replace(/\\n/g, '\n').replace(/\\"/g, '"');
              const stripped = maybeStripReasoning(decoded);
              const reencoded = stripped.replace(/\n/g, '\\n').replace(/"/g, '\\"');
              return `"content":"${reencoded}"`;
            }
          );
          res.write(cleaned);
        } else {
          res.write(raw);
        }
      });

      nimResponse.data.on('end', () => res.end());
      nimResponse.data.on('error', (err) => {
        console.error('Stream error:', err.message);
        res.end();
      });

      return;
    }

    // ── Non-streaming response ─────────────────────────────────────────────
    const data = nimResponse.data;

    if (!SHOW_REASONING && data.choices) {
      data.choices = data.choices.map((choice) => {
        if (choice.message?.content) {
          choice.message.content = maybeStripReasoning(choice.message.content);
        }
        return choice;
      });
    }

    // Add a helpful header so the caller knows which NIM model was used
    res.setHeader('X-NIM-Model', nimModel);
    res.setHeader('X-NIM-Model-Label', meta.label || nimModel);
    res.json(data);

  } catch (err) {
    const status  = err.response?.status || 500;
    const nimError = err.response?.data;

    console.error(`[${new Date().toISOString()}] NIM API error (HTTP ${status}):`, nimError || err.message);

    res.status(status).json({
      error: {
        message: nimError?.detail || nimError?.message || err.message || 'Unknown NIM API error',
        type: 'nim_api_error',
        status,
        nim_error: nimError || null,
      },
    });
  }
});

// Embeddings proxy (passthrough)
app.post('/v1/embeddings', async (req, res) => {
  if (!NIM_API_KEY) {
    return res.status(500).json({ error: { message: 'NIM_API_KEY not configured.' } });
  }

  try {
    const response = await axios.post(`${NIM_API_BASE}/embeddings`, req.body, {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json',
      },
      timeout: 30_000,
    });
    res.json(response.data);
  } catch (err) {
    const status = err.response?.status || 500;
    res.status(status).json({ error: { message: err.message, status } });
  }
});

// 404 catch-all
app.use((req, res) => {
  res.status(404).json({
    error: { message: `Route not found: ${req.method} ${req.path}`, type: 'not_found' },
  });
});

// ─── Start ───────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`
╔══════════════════════════════════════════════════════════╗
║         NVIDIA NIM Proxy — OpenAI-Compatible API         ║
╠══════════════════════════════════════════════════════════╣
║  Listening on  : http://localhost:${String(PORT).padEnd(26)}║
║  NIM base URL  : ${NIM_API_BASE.substring(0, 37).padEnd(38)}║
║  API key       : ${NIM_API_KEY ? '✅ configured' : '❌ NOT SET — set NIM_API_KEY'.padEnd(38)}${NIM_API_KEY ? '                         ' : ''}║
║  Show reasoning: ${String(SHOW_REASONING).padEnd(38)}║
║  Thinking mode : ${String(ENABLE_THINKING_MODE).padEnd(38)}║
║  Models loaded : ${String(Object.keys(NIM_MODELS).length).padEnd(38)}║
╚══════════════════════════════════════════════════════════╝

Available NIM models:
${Object.entries(NIM_MODELS)
  .map(([id, m]) => `  • ${id.padEnd(48)} ${m.multimodal ? '🖼 ' : '   '}${m.thinking ? '🧠' : '  '}  ${m.label}`)
  .join('\n')}

Endpoints:
  GET  /health              — Health check
  GET  /v1/models           — List all models
  POST /v1/chat/completions — Chat (streaming & non-streaming)
  POST /v1/embeddings       — Embeddings passthrough
`);
});    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      try {
        await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model: model,
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1
        }, {
          headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          validateStatus: (status) => status < 500
        }).then(res => {
          if (res.status >= 200 && res.status < 300) {
            nimModel = model;
          }
        });
      } catch (e) {}
      
      if (!nimModel) {
        const modelLower = model.toLowerCase();
        if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
          nimModel = 'meta/llama-3.1-405b-instruct';
        } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
          nimModel = 'meta/llama-3.1-70b-instruct';
        } else {
          nimModel = 'meta/llama-3.1-8b-instruct';
        }
      }
    }
    
    // Transform OpenAI request to NIM format
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 9024,
      extra_body: ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : undefined,
      stream: stream || false
    };
    
    // Make request to NVIDIA NIM API
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });
    
    if (stream) {
      // Handle streaming response with reasoning
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\\n');
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;
                
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  
                  if (content && reasoningStarted) {
                    combinedContent += '</think>\\n\\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  
                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                  }
                } else {
                  if (content) {
                    data.choices[0].delta.content = content;
                  } else {
                    data.choices[0].delta.content = '';
                  }
                  delete data.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\\n\\n`);
            } catch (e) {
              res.write(line + '\\n');
            }
          }
        });
      });
      
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
      // Transform NIM response to OpenAI format with reasoning
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\\n' + choice.message.reasoning_content + '\\n</think>\\n\\n' + fullContent;
          }
          
          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('Proxy error:', error.message);
    
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
