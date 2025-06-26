🛡️ Content Moderation API

A FastAPI service for automated text moderation: detects PII (Russian/English) and toxic content. Plug & play with Dify or any modern platform.

🚀 Quick Start

Clone and run with Docker Compose:

git clone https://github.com/Aggregain/content-moderation.git

cd content-moderation

docker-compose up --build

API: http://localhost:8000/api/dify/moderation

🔌 API Usage

POST /api/dify/moderation

Headers:

Content-Type: application/json

Authorization: Bearer 123456

Request Example:

{
  "point": "app.moderation.input",
  "params": {
    "inputs": { "text": "your message", "lang": "ru" },
    "query": "your message"
  }
}
Response Example:

{
  "flagged": true,
  "action": "direct_output",
  "preset_response": "В вашем сообщении обнаружены персональные данные и высокая токсичность."
}


✨ Features

🔍 Detects PII (names, SNILS, phone, email)

🇷🇺 🇬🇧 Russian and English support

☣️ Toxicity check (TextDetox)

🐳 Docker & Dify ready

📝 Notes

Place pattern files (russian_pii_patterns.xlsx, rus_phone_regions.xlsx) in the root folder.

Use a fast network or PyPI mirror for smooth Docker builds.

🪪 License: MIT
