// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = false; // Set to true to enable chat_template_kwargs thinking parameter

// Model mapping (adjust based on available NIM models)
// Model mapping (adjust based on available NIM models)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking',

  // ── Nuevos modelos ──────────────────────────────────────────────────────
  // DeepSeek V3.2 — 685B MoE, razonamiento avanzado + agentes (dic 2025)
  'o1':             'deepseek-ai/deepseek-v3.2',

  // Kimi K2.5 — 1T multimodal MoE, visión + texto + thinking/instant mode
  'o1-mini':        'moonshotai/kimi-k2.5',

  // Kimi K2 Thinking — reasoning puro, 256K contexto, INT4 nativo
  'o3':             'moonshotai/kimi-k2-thinking',

  // Kimi K2 Instruct 0905 — versión más reciente de K2 con 256K contexto
  'o3-mini':        'moonshotai/kimi-k2-instruct-0905',

  // Qwen3-235B-A22B — MoE de última generación, thinking/non-thinking mode
  'gpt-4.5':        'qwen/qwen3-235b-a22b',

  // Qwen3.5-397B-A17B — VLM multimodal (texto + imagen + vídeo), feb 2026
  'gpt-4.5-turbo':  'qwen/qwen3.5-397b-a17b',
};

// Health check endpoint
app.get('/heal
th', (req, res) => {
  res.json({
