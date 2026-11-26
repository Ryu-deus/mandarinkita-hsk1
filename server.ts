import 'dotenv/config';
import express, { Request, Response } from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI, Type } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = Number(process.env.PORT) || 3000;

app.use(express.json({ limit: '1mb' }));

interface ChatHistoryItem {
  role: 'user' | 'assistant';
  content: string;
}

interface TutorRequestBody {
  message?: string;
  scenario?: string;
  history?: ChatHistoryItem[];
}

function getAiClient(): GoogleGenAI | null {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') {
    return null;
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

app.post('/api/tutor/chat', async (req: Request, res: Response) => {
  const body = (req.body ?? {}) as TutorRequestBody;
  const userMessage = typeof body.message === 'string' ? body.message.trim() : '';
  const scenario = typeof body.scenario === 'string' ? body.scenario.trim() : '';
  const history = Array.isArray(body.history) ? body.history.slice(-8) : [];

  if (!userMessage) {
    res.status(400).json({ error: 'Pesan tidak boleh kosong.' });
    return;
  }

  const ai = getAiClient();
  if (!ai) {
    res.status(503).json({
      error:
        'Kunci API Gemini belum dikonfigurasi pada server. Silakan periksa pengaturan Secrets pada panel AI Studio.',
    });
    return;
  }

  const systemInstruction = `Kamu adalah Laoshi (guru) bahasa Mandarin yang ramah, sabar, dan suportif untuk pemula penutur bahasa Indonesia yang sedang mempelajari tingkat HSK 1.

Aturan wajib untuk setiap balasan:
1. Gunakan HANYA kosakata dan struktur tata bahasa tingkat HSK 1 (kalimat pendek, sederhana, jelas, mudah dipahami pemula).
2. Jika pengguna menulis dalam bahasa Indonesia, Mandarin (Hanzi), atau Pinyin, balas dengan percakapan yang natural sesuai konteks skenario${scenario ? ` "${scenario}"` : ''}.
3. Isi field "hanzi" hanya dengan karakter Mandarin sederhana (Simplified Chinese) yang sesuai tingkat HSK 1.
4. Isi field "pinyin" dengan Pinyin lengkap bertanda nada (diakritik seperti ā, á, ǎ, à) yang persis cocok dengan "hanzi".
5. Isi field "translation" dengan terjemahan bahasa Indonesia yang natural dan akurat dari kalimat Mandarin tersebut.
6. Isi field "feedback" dalam bahasa Indonesia yang hangat:
   - Jika pengguna membuat kesalahan tata bahasa, pemilihan kata, atau pinyin, berikan koreksi yang ramah dan jelas beserta bentuk yang benar.
   - Jika pengguna sudah benar atau baru memulai, berikan apresiasi singkat dan 1 tips kosakata/pengucapan HSK 1 yang relevan.
7. Isi field "followUpSuggestion" dengan 1 contoh kalimat balasan pendek dalam format "Hanzi (Pinyin) — Arti Indonesia" yang bisa dicoba pengguna selanjutnya.`;

  const conversationTranscript = history
    .map((item) => `${item.role === 'user' ? 'Murid' : 'Laoshi'}: ${item.content}`)
    .join('\n');

  const prompt = conversationTranscript
    ? `Riwayat percakapan sebelumnya:\n${conversationTranscript}\n\nPesan terbaru murid: ${userMessage}`
    : `Pesan murid: ${userMessage}`;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-4-hydrogen',
      contents: prompt,
      config: {
        systemInstruction,
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            hanzi: {
              type: Type.STRING,
              description: 'Kalimat balasan Mandarin HSK 1 dalam aksara Hanzi sederhana.',
            },
            pinyin: {
              type: Type.STRING,
              description: 'Pinyin lengkap dengan tanda nada untuk kalimat Hanzi di atas.',
            },
            translation: {
              type: Type.STRING,
              description: 'Terjemahan bahasa Indonesia dari kalimat Hanzi di atas.',
            },
            feedback: {
              type: Type.STRING,
              description:
                'Koreksi ramah jika ada kesalahan atau catatan belajar singkat dalam bahasa Indonesia.',
            },
            followUpSuggestion: {
              type: Type.STRING,
              description:
                'Satu saran balasan pendek HSK 1 untuk dicoba murid berikutnya.',
            },
          },
          required: ['hanzi', 'pinyin', 'translation', 'feedback'],
        },
      },
    });

    const rawText = (response.text ?? '').trim();
    if (!rawText) {
      res.status(502).json({ error: 'Laoshi AI tidak memberikan respons. Silakan coba lagi.' });
      return;
    }

    const parsed = JSON.parse(rawText);
    res.json({
      reply: {
        hanzi: String(parsed.hanzi ?? ''),
        pinyin: String(parsed.pinyin ?? ''),
        translation: String(parsed.translation ?? ''),
        feedback: String(parsed.feedback ?? ''),
        followUpSuggestion: parsed.followUpSuggestion
          ? String(parsed.followUpSuggestion)
          : undefined,
      },
    });
  } catch (err: unknown) {
    const status =
      typeof err === 'object' && err !== null && 'status' in err
        ? Number((err as { status?: unknown }).status)
        : 500;
    const message = err instanceof Error ? err.message : 'Terjadi gangguan pada layanan AI.';
    console.error('Error calling Gemini API:', message);

    if (status === 400 || status === 403 || message.includes('API_KEY') || message.includes('PERMISSION_DENIED')) {
      res.status(403).json({
        error:
          'Kunci API Gemini belum aktif atau tidak memiliki izin. Silakan periksa GEMINI_API_KEY di panel Settings > Secrets.',
      });
      return;
    }

    if (status === 429 || message.includes('RESOURCE_EXHAUSTED')) {
      res.status(429).json({
        error:
          'Batas kuota permintaan AI sementara tercapai. Silakan coba lagi sebentar atau periksa kuota API key di Settings > Secrets.',
      });
      return;
    }

    res.status(500).json({
      error: 'Maaf, Laoshi AI sedang mengalami gangguan koneksi. Silakan coba beberapa saat lagi.',
    });
  }
});

async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`MandarinKita HSK 1 server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
