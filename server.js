const express = require('express');
const multer = require('multer');
const cors = require('cors');
const axios = require('axios');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Verify API keys are loaded
console.log('🔑 API Keys Check:');
console.log('   Google Cloud API Key:', process.env.GOOGLE_CLOUD_API_KEY ? '✅ Loaded' : '❌ Missing');
console.log('   Groq API Key:', process.env.GROK_API_KEY ? '✅ Loaded' : '❌ Missing');

// ===== CORS CONFIGURATION =====
const corsOptions = {
  origin: process.env.FRONTEND_URL || '*',
  credentials: true,
  optionsSuccessStatus: 200,
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
};

app.use(cors(corsOptions));
app.use(express.json());

// ===== MULTER - MEMORY STORAGE (No Disk Storage) =====
const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 } // 50MB limit
});

// Image classification endpoint
app.post('/api/classify', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image uploaded' });
    }

    console.log(`\n📸 Processing image: ${req.file.originalname}`);

    // Convert buffer directly to base64 (no file saved to disk)
    const base64Image = req.file.buffer.toString('base64');

    // Step 1: Google Cloud Vision - Web Detection to identify artwork
    console.log('🌐 Using Web Detection to identify artwork...');
    const artworkName = await webDetection(base64Image);
    console.log(`✅ Identified artwork: ${artworkName}`);

    // Step 2: Get additional vision data
    const labels = await labelDetection(base64Image);
    const objects = await objectLocalization(base64Image);
    const detectedText = await textDetection(base64Image);

    // Step 3: Groq Cloud - Generate detailed artwork description
    console.log(`🤖 Generating artwork details with Groq for: ${artworkName}`);
    const groqDescription = await generateArtworkDescription(artworkName);

    // Send response
    res.json({
      status: 'success',
      timestamp: new Date().toISOString(),
      detectedArtwork: artworkName,
      visionAnalysis: {
        labels: labels.slice(0, 10),
        objects: objects.slice(0, 10),
        detectedText: detectedText.substring(0, 500)
      },
      artworkDetails: groqDescription
    });

  } catch (error) {
    console.error('❌ Error:', error.message);
    res.status(500).json({ 
      error: 'Image classification failed', 
      details: error.message 
    });
  }
});

// Google Cloud Vision - Web Detection (to identify painting name)
async function webDetection(base64Image) {
  try {
    const response = await axios.post(
      `https://vision.googleapis.com/v1/images:annotate?key=${process.env.GOOGLE_CLOUD_API_KEY}`,
      {
        requests: [
          {
            image: {
              content: base64Image
            },
            features: [
              {
                type: 'WEB_DETECTION',
                maxResults: 10
              }
            ]
          }
        ]
      }
    );

    const webDetectionData = response.data.responses[0].webDetection;
    
    if (webDetectionData && webDetectionData.webEntities && webDetectionData.webEntities.length > 0) {
      const relevantEntity = webDetectionData.webEntities
        .filter(entity => entity.description && entity.score > 0.5)
        .sort((a, b) => (b.score || 0) - (a.score || 0))[0];
      
      if (relevantEntity && relevantEntity.description) {
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
            image: {
              content: base64Image
            },
            features: [
              {
                type: 'LABEL_DETECTION',
                maxResults: 10
              }
            ]
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
            image: {
              content: base64Image
            },
            features: [
              {
                type: 'OBJECT_LOCALIZATION',
                maxResults: 10
              }
            ]
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
            image: {
              content: base64Image
            },
            features: [
              {
                type: 'TEXT_DETECTION'
              }
            ]
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
      console.error('❌ GROK_API_KEY not found in environment');
      return { 
        title: artworkName,
        error: 'Groq API key not configured'
      };
    }

    console.log('📤 Sending request to Groq API...');
    console.log('   Model: llama-3.3-70b-versatile');
    console.log('   Artwork: ' + artworkName);

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
      const parsed = jsonMatch ? JSON.parse(jsonMatch[0]) : null;
      
      if (parsed) {
        return parsed;
      } else {
        return { 
          title: artworkName, 
          description: content 
        };
      }
    } catch (e) {
      console.log('⚠️ Failed to parse JSON response');
      return { 
        title: artworkName, 
        description: content 
      };
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
  console.log(`\n🚀 Server running on http://localhost:${PORT}`);
  console.log(`📊 Classification endpoint: POST http://localhost:${PORT}/api/classify`);
  console.log(`💚 Health check: GET http://localhost:${PORT}/api/health`);
});
