const express = require('express');
const multer = require('multer');
const axios = require('axios');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Verify API keys are loaded
console.log('🔑 API Keys Check:');
console.log('   Google Cloud API Key:', process.env.GOOGLE_CLOUD_API_KEY ? '✅ Loaded' : '❌ Missing');
console.log('   Groq API Key:', process.env.GROK_API_KEY ? '✅ Loaded' : '❌ Missing');

// ===== EXPLICIT CORS HEADERS =====
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
  
  if (req.method === 'OPTIONS') {
    return res.sendStatus(200);
  }
  
  next();
});

app.use(express.json());

// ===== MULTER - MEMORY STORAGE =====
const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 }
});

// Shared image processing logic
async function processImage(base64Image, artworkName) {
  const labels = await labelDetection(base64Image);
  const objects = await objectLocalization(base64Image);
  const detectedText = await textDetection(base64Image);
  const groqDescription = await generateArtworkDescription(artworkName);

  return {
    status: 'success',
    timestamp: new Date().toISOString(),
    detectedArtwork: artworkName,
    visionAnalysis: {
      labels: labels.slice(0, 10),
      objects: objects.slice(0, 10),
      detectedText: detectedText.substring(0, 500)
    },
    artworkDetails: groqDescription
  };
}

// ===== ENDPOINT: /api/scan/vision (Frontend calls this) =====
app.post('/api/scan/vision', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ 
        status: 'error',
        error: 'No image uploaded' 
      });
    }

    console.log(`\n📸 Vision API: Processing ${req.file.originalname}`);

    const base64Image = req.file.buffer.toString('base64');

    console.log('🌐 Web Detection...');
    const artworkName = await webDetection(base64Image);
    console.log(`✅ Identified: ${artworkName}`);

    const result = await processImage(base64Image, artworkName);
    res.json(result);

  } catch (error) {
    console.error('❌ Vision API Error:', error.message);
    res.status(500).json({ 
      status: 'error',
      error: 'Image classification failed', 
      details: error.message 
    });
  }
});

// ===== ENDPOINT: /api/classify (Alternative endpoint) =====
app.post('/api/classify', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image uploaded' });
    }

    console.log(`\n📸 Classify API: Processing ${req.file.originalname}`);

    const base64Image = req.file.buffer.toString('base64');

    console.log('🌐 Web Detection...');
    const artworkName = await webDetection(base64Image);
    console.log(`✅ Identified: ${artworkName}`);

    const result = await processImage(base64Image, artworkName);
    res.json(result);

  } catch (error) {
    console.error('❌ Classify API Error:', error.message);
    res.status(500).json({ 
      error: 'Image classification failed', 
      details: error.message 
    });
  }
});

// Google Cloud Vision - Web Detection
async function webDetection(base64Image) {
  try {
    const response = await axios.post(
      `https://vision.googleapis.com/v1/images:annotate?key=${process.env.GOOGLE_CLOUD_API_KEY}`,
      {
        requests: [
          {
            image: { content: base64Image },
            features: [{ type: 'WEB_DETECTION', maxResults: 10 }]
          }
        ]
      }
    );

    const webDetectionData = response.data.responses[0].webDetection;
    
    if (webDetectionData?.webEntities?.length > 0) {
      const relevantEntity = webDetectionData.webEntities
        .filter(entity => entity.description && entity.score > 0.5)
        .sort((a, b) => (b.score || 0) - (a.score || 0))[0];
      
      if (relevantEntity?.description) {
        return relevantEntity.description;
      }
    }

    const labels = await labelDetection(base64Image);
    return labels.length > 0 ? labels[0].description : 'Unknown Artwork';

  } catch (error) {
    console.error('❌ Web Detection Error:', error.message);
    return 'Unknown Artwork';
  }
}

// Google Cloud Vision - Label Detection
async function labelDetection(base64Image) {
  try {
    const response = await axios.post(
      `https://vision.googleapis.com/v1/images:annotate?key=${process.env.GOOGLE_CLOUD_API_KEY}`,
      {
        requests: [
          {
            image: { content: base64Image },
            features: [{ type: 'LABEL_DETECTION', maxResults: 10 }]
          }
        ]
      }
    );

    return response.data.responses[0].labelAnnotations?.map(label => ({
      description: label.description,
      confidence: (label.score * 100).toFixed(2)
    })) || [];
  } catch (error) {
    console.error('❌ Label Detection Error:', error.message);
    return [];
  }
}

// Google Cloud Vision - Object Localization
async function objectLocalization(base64Image) {
  try {
    const response = await axios.post(
      `https://vision.googleapis.com/v1/images:annotate?key=${process.env.GOOGLE_CLOUD_API_KEY}`,
      {
        requests: [
          {
            image: { content: base64Image },
            features: [{ type: 'OBJECT_LOCALIZATION', maxResults: 10 }]
          }
        ]
      }
    );

    return response.data.responses[0].localizedObjectAnnotations?.map(obj => ({
      name: obj.name,
      confidence: (obj.score * 100).toFixed(2)
    })) || [];
  } catch (error) {
    console.error('❌ Object Localization Error:', error.message);
    return [];
  }
}

// Google Cloud Vision - Text Detection
async function textDetection(base64Image) {
  try {
    const response = await axios.post(
      `https://vision.googleapis.com/v1/images:annotate?key=${process.env.GOOGLE_CLOUD_API_KEY}`,
      {
        requests: [
          {
            image: { content: base64Image },
            features: [{ type: 'TEXT_DETECTION' }]
          }
        ]
      }
    );

    return response.data.responses[0].textAnnotations?.[0]?.description || 'No text detected';
  } catch (error) {
    console.error('❌ Text Detection Error:', error.message);
    return 'No text detected';
  }
}

// Groq Cloud - Generate Artwork Description
async function generateArtworkDescription(artworkName) {
  try {
    const apiKey = process.env.GROK_API_KEY;
    
    if (!apiKey) {
      console.error('❌ GROK_API_KEY not found');
      return { 
        title: artworkName,
        error: 'Groq API key not configured'
      };
    }

    console.log('📤 Groq: Generating description...');

    const response = await axios.post(
      'https://api.groq.com/openai/v1/chat/completions',
      {
        model: 'llama-3.3-70b-versatile',
        messages: [{
          role: 'user',
          content: `Tell me about "${artworkName}" in JSON format with exactly these fields: title, artist, year_created, description, historical_context, artistic_technique, significance. If it's a famous artwork, provide accurate information. Format response as valid JSON only.`,
        }],
        temperature: 0.7,
        max_tokens: 1500,
      },
      {
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        timeout: 30000,
      }
    );

    const content = response.data.choices[0]?.message?.content;

    if (!content) {
      throw new Error('No content from Groq');
    }

    console.log('✅ Groq response received');

    try {
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      return jsonMatch ? JSON.parse(jsonMatch[0]) : { title: artworkName, description: content };
    } catch (e) {
      console.log('⚠️ Failed to parse JSON, returning as text');
      return { title: artworkName, description: content };
    }
  } catch (error) {
    console.error('❌ Groq Error:', error.response?.status, error.message);
    
    if (error.response?.status === 401) {
      return {
        title: artworkName,
        error: 'Groq authentication failed - invalid API key',
        status: 401
      };
    }
    
    return {
      title: artworkName,
      error: 'Failed to generate description',
      details: error.message
    };
  }
}

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'Server is running',
    timestamp: new Date().toISOString(),
    apiKeysConfigured: {
      googleCloud: !!process.env.GOOGLE_CLOUD_API_KEY,
      groq: !!process.env.GROK_API_KEY
    }
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('❌ Unhandled error:', err);
  res.status(500).json({ 
    error: 'Internal server error',
    details: err.message 
  });
});

app.listen(PORT, () => {
  console.log(`\n🚀 Server running on port ${PORT}`);
  console.log(`📊 Vision API: POST /api/scan/vision (Frontend)`);
  console.log(`📊 Classify API: POST /api/classify (Curl/Testing)`);
  console.log(`💚 Health: GET /api/health`);
});
