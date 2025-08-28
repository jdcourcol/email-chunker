# Admin API for Service Administrators

A comprehensive REST API that allows service administrators to manage any entity within any project in the email/document system.

## 🚀 Features

- **🔐 Secure Authentication**: JWT-based authentication with role-based access control
- **📧 Email Management**: Full CRUD operations on email entities
- **📄 PDF Management**: Full CRUD operations on PDF document entities
- **🔧 System Maintenance**: Database optimization, cleanup, and embedding recomputation
- **📊 System Monitoring**: Real-time system status and statistics
- **⚡ Bulk Operations**: Efficient bulk updates for multiple entities
- **🛡️ Admin-Only Access**: Restricted to service administrators only

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Admin Client  │    │   Admin API     │    │   Database      │
│                 │    │                 │    │                 │
│ • Authentication│◄──►│ • Flask Server  │◄──►│ • PostgreSQL   │
│ • CRUD Ops      │    │ • JWT Auth      │    │ • pgvector      │
│ • Bulk Updates  │    │ • Admin Routes  │    │ • Embeddings    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 API Endpoints

### Authentication

#### `POST /auth/login`
Authenticate as a service administrator.

**Request:**
```json
{
  "username": "admin",
  "password": "admin123"
}
```

**Response:**
```json
{
  "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "user": {
    "username": "admin",
    "role": "service_admin",
    "permissions": ["read", "write", "delete", "admin"]
  }
}
```

### Entity Management

#### `GET /admin/entities/{entity_type}`
List all entities of a specific type.

**Supported entity types:**
- `emails` - Email messages
- `pdfs` - PDF documents
- `projects` - Project metadata (future)

**Headers:**
```
Authorization: Bearer <jwt_token>
```

**Response:**
```json
{
  "entity_type": "emails",
  "count": 150,
  "entities": [...]
}
```

#### `GET /admin/entities/{entity_type}/{entity_id}`
Get a specific entity by ID.

**Response:**
```json
{
  "entity_type": "emails",
  "entity": {
    "id": 123,
    "subject": "Meeting Tomorrow",
    "sender": "john@example.com",
    "recipient": "team@example.com",
    "folder": "INBOX",
    "date_sent": "2025-01-15T10:30:00Z",
    "body": "Let's meet tomorrow at 2 PM...",
    "content_type": "text/plain"
  }
}
```

#### `PUT /admin/entities/{entity_type}/{entity_id}`
Update a specific entity.

**Request:**
```json
{
  "subject": "Updated Meeting Subject",
  "folder": "WORK",
  "body": "Updated meeting details..."
}
```

**Response:**
```json
{
  "message": "Email updated successfully",
  "entity_id": 123
}
```

#### `DELETE /admin/entities/{entity_type}/{entity_id}`
Delete a specific entity.

**Response:**
```json
{
  "message": "Email deleted successfully",
  "entity_id": 123
}
```

### Bulk Operations

#### `POST /admin/bulk-update/{entity_type}`
Bulk update multiple entities of the same type.

**Request:**
```json
{
  "updates": [
    {
      "id": 123,
      "data": {
        "subject": "Updated Subject 1",
        "folder": "WORK"
      }
    },
    {
      "id": 124,
      "data": {
        "subject": "Updated Subject 2",
        "folder": "PERSONAL"
      }
    }
  ]
}
```

**Response:**
```json
{
  "entity_type": "emails",
  "total_updates": 2,
  "results": [
    {
      "id": 123,
      "success": true,
      "error": null
    },
    {
      "id": 124,
      "success": true,
      "error": null
    }
  ]
}
```

### System Management

#### `GET /admin/system/status`
Get system status and statistics.

**Response:**
```json
{
  "database": {
    "connected": true,
    "tables": [
      {
        "name": "emails",
        "record_count": 150
      },
      {
        "name": "pdf_documents",
        "record_count": 25
      }
    ]
  },
  "entities": {
    "emails": 150,
    "pdfs": 25
  },
  "system": {
    "timestamp": "2025-01-15T10:30:00Z",
    "version": "1.0.0"
  }
}
```

#### `POST /admin/system/maintenance`
Perform system maintenance tasks.

**Available actions:**
- `recompute_embeddings` - Recompute all embeddings
- `cleanup_orphaned` - Clean up orphaned records
- `optimize_database` - Optimize database tables

**Request:**
```json
{
  "action": "cleanup_orphaned"
}
```

**Response:**
```json
{
  "action": "cleanup_orphaned",
  "success": true,
  "message": "Orphaned records cleanup completed"
}
```

## 🔐 Security

### Authentication Flow

1. **Login**: Admin provides credentials via `/auth/login`
2. **JWT Token**: Server returns a JWT token valid for 24 hours
3. **Authorization**: Client includes token in `Authorization: Bearer <token>` header
4. **Validation**: Server validates token and checks admin role

### Access Control

- **Service Admin Role**: Required for all admin endpoints
- **JWT Expiration**: Tokens expire after 24 hours
- **Secure Headers**: All requests must include valid JWT token

### Production Security

⚠️ **Important**: Change default secrets in production:

```python
app.config['SECRET_KEY'] = 'your-production-secret-key'
app.config['JWT_SECRET_KEY'] = 'your-production-jwt-secret-key'
```

## 🚀 Getting Started

### 1. Install Dependencies

```bash
uv add flask flask-cors pyjwt werkzeug
```

### 2. Start the Admin API Server

```bash
python admin_api.py
```

The server will start on `http://localhost:8000`

### 3. Use the Admin Client

```bash
python admin_client.py
```

### 4. Manual API Testing

#### Login
```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```

#### List Emails (with token)
```bash
curl -X GET http://localhost:8000/admin/entities/emails \
  -H "Authorization: Bearer <your_jwt_token>"
```

## 📝 Usage Examples

### Python Client

```python
from admin_client import AdminClient

# Initialize client
client = AdminClient("http://localhost:8000")

# Login
if client.login("admin", "admin123"):
    # Get system status
    status = client.get_system_status()
    print(f"Total emails: {status['entities']['emails']}")
    
    # List emails
    emails = client.list_entities("emails")
    
    # Update an email
    if emails:
        email_id = emails[0]['id']
        client.update_entity("emails", email_id, {
            "subject": "Updated Subject",
            "folder": "WORK"
        })
```

### JavaScript/Node.js

```javascript
const axios = require('axios');

class AdminClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
        this.token = null;
    }
    
    async login(username, password) {
        const response = await axios.post(`${this.baseUrl}/auth/login`, {
            username, password
        });
        this.token = response.data.token;
        return response.data;
    }
    
    async listEntities(entityType) {
        const response = await axios.get(
            `${this.baseUrl}/admin/entities/${entityType}`,
            { headers: { Authorization: `Bearer ${this.token}` } }
        );
        return response.data;
    }
}

// Usage
const client = new AdminClient();
client.login('admin', 'admin123')
    .then(() => client.listEntities('emails'))
    .then(data => console.log(`Found ${data.count} emails`));
```

## 🔧 Configuration

### Environment Variables

```bash
export FLASK_SECRET_KEY="your-secret-key"
export JWT_SECRET_KEY="your-jwt-secret-key"
export DB_HOST="localhost"
export DB_PORT="5432"
export DB_NAME="email_archive"
export DB_USER="admin"
export DB_PASSWORD="password"
```

### Database Setup

Ensure your PostgreSQL database has the required tables:

```sql
-- Emails table
CREATE TABLE IF NOT EXISTS emails (
    id SERIAL PRIMARY KEY,
    message_id VARCHAR(500) UNIQUE,
    subject TEXT,
    sender VARCHAR(500),
    recipient VARCHAR(500),
    date_sent TIMESTAMP,
    folder VARCHAR(100),
    content_type VARCHAR(100),
    body TEXT,
    headers JSONB,
    attachments JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- PDF documents table
CREATE TABLE IF NOT EXISTS pdf_documents (
    id SERIAL PRIMARY KEY,
    file_path VARCHAR(1000),
    file_name VARCHAR(500),
    folder_name VARCHAR(100),
    content TEXT,
    content_length INTEGER,
    page_count INTEGER,
    file_size BIGINT,
    title VARCHAR(500),
    author VARCHAR(500),
    subject VARCHAR(500),
    creator VARCHAR(500),
    producer VARCHAR(500),
    created_date TIMESTAMP,
    modified_date TIMESTAMP,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🧪 Testing

### Run the Demo

```bash
python admin_client.py
```

### Test Individual Endpoints

```bash
# Test authentication
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# Test protected endpoint (replace <token> with actual token)
curl -X GET http://localhost:8000/admin/system/status \
  -H "Authorization: Bearer <token>"
```

## 🚨 Error Handling

The API returns appropriate HTTP status codes:

- `200` - Success
- `400` - Bad Request (invalid data)
- `401` - Unauthorized (missing/invalid token)
- `403` - Forbidden (insufficient privileges)
- `404` - Not Found (entity doesn't exist)
- `500` - Internal Server Error

Error responses include descriptive messages:

```json
{
  "error": "Entity not found",
  "details": "Email with ID 999 does not exist"
}
```

## 🔮 Future Enhancements

- **Project Management**: Full project CRUD operations
- **User Management**: Admin user creation and management
- **Audit Logging**: Track all admin actions
- **Rate Limiting**: Prevent API abuse
- **Web Interface**: Admin dashboard
- **Batch Operations**: Large-scale data operations
- **Backup/Restore**: Database backup and restoration

## 📞 Support

For issues or questions:

1. Check the logs in the admin API server
2. Verify database connectivity
3. Ensure proper authentication
4. Check entity type and ID validity

## 📄 License

This admin API is part of the Email Chunker project and follows the same MIT license.

