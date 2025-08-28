#!/usr/bin/env python3
"""
Admin API for Service Administrators

Provides update endpoints for service administrators to manage any entity
within any project in the email/document system.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from werkzeug.security import check_password_hash, generate_password_hash
import jwt
from functools import wraps

# Import our existing modules
from database_manager import DatabaseManager
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change in production
app.config['JWT_SECRET_KEY'] = 'your-jwt-secret-key-here'  # Change in production
CORS(app)

# Global database manager
db_manager = None

# Admin users (in production, this should be in a database)
ADMIN_USERS = {
    'admin': {
        'password_hash': generate_password_hash('admin123'),
        'role': 'service_admin',
        'permissions': ['read', 'write', 'delete', 'admin']
    }
}

def require_auth(f):
    """Decorator to require authentication for protected endpoints."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'error': 'No authorization token provided'}), 401
        
        try:
            # Remove 'Bearer ' prefix if present
            if token.startswith('Bearer '):
                token = token[7:]
            
            # Verify token
            payload = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
            request.current_user = payload
            return f(*args, **kwargs)
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
    
    return decorated

def require_admin(f):
    """Decorator to require admin privileges."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not hasattr(request, 'current_user'):
            return jsonify({'error': 'Authentication required'}), 401
        
        if request.current_user.get('role') != 'service_admin':
            return jsonify({'error': 'Admin privileges required'}), 403
        
        return f(*args, **kwargs)
    
    return decorated

@app.route('/auth/login', methods=['POST'])
def login():
    """Admin login endpoint."""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'error': 'Username and password required'}), 400
        
        user = ADMIN_USERS.get(username)
        if not user or not check_password_hash(user['password_hash'], password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Generate JWT token
        token = jwt.encode({
            'username': username,
            'role': user['role'],
            'permissions': user['permissions'],
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, app.config['JWT_SECRET_KEY'], algorithm='HS256')
        
        return jsonify({
            'token': token,
            'user': {
                'username': username,
                'role': user['role'],
                'permissions': user['permissions']
            }
        })
    
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/admin/entities/<entity_type>', methods=['GET'])
@require_auth
@require_admin
def list_entities(entity_type: str):
    """List all entities of a specific type."""
    try:
        if entity_type == 'emails':
            entities = db_manager.get_all_emails()
        elif entity_type == 'pdfs':
            entities = db_manager.get_all_pdf_documents()
        elif entity_type == 'projects':
            entities = db_manager.get_all_projects()
        else:
            return jsonify({'error': f'Unknown entity type: {entity_type}'}), 400
        
        return jsonify({
            'entity_type': entity_type,
            'count': len(entities),
            'entities': entities
        })
    
    except Exception as e:
        logger.error(f"Error listing {entity_type}: {e}")
        return jsonify({'error': f'Failed to list {entity_type}'}), 500

@app.route('/admin/entities/<entity_type>/<int:entity_id>', methods=['GET'])
@require_auth
@require_admin
def get_entity(entity_type: str, entity_id: int):
    """Get a specific entity by ID."""
    try:
        if entity_type == 'emails':
            entity = db_manager.get_email_by_id(entity_id)
        elif entity_type == 'pdfs':
            entity = db_manager.get_pdf_document_by_id(entity_id)
        elif entity_type == 'projects':
            entity = db_manager.get_project_by_id(entity_id)
        else:
            return jsonify({'error': f'Unknown entity type: {entity_type}'}), 400
        
        if not entity:
            return jsonify({'error': f'{entity_type.capitalize()} not found'}), 404
        
        return jsonify({
            'entity_type': entity_type,
            'entity': entity
        })
    
    except Exception as e:
        logger.error(f"Error getting {entity_type} {entity_id}: {e}")
        return jsonify({'error': f'Failed to get {entity_type}'}), 500

@app.route('/admin/entities/<entity_type>/<int:entity_id>', methods=['PUT'])
@require_auth
@require_admin
def update_entity(entity_type: str, entity_id: int):
    """Update a specific entity."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No update data provided'}), 400
        
        success = False
        
        if entity_type == 'emails':
            success = db_manager.update_email(entity_id, data)
        elif entity_type == 'pdfs':
            success = db_manager.update_pdf_document(entity_id, data)
        elif entity_type == 'projects':
            success = db_manager.update_project(entity_id, data)
        else:
            return jsonify({'error': f'Unknown entity type: {entity_type}'}), 400
        
        if success:
            return jsonify({
                'message': f'{entity_type.capitalize()} updated successfully',
                'entity_id': entity_id
            })
        else:
            return jsonify({'error': f'Failed to update {entity_type}'}), 500
    
    except Exception as e:
        logger.error(f"Error updating {entity_type} {entity_id}: {e}")
        return jsonify({'error': f'Failed to update {entity_type}'}), 500

@app.route('/admin/entities/<entity_type>/<int:entity_id>', methods=['DELETE'])
@require_auth
@require_admin
def delete_entity(entity_type: str, entity_id: int):
    """Delete a specific entity."""
    try:
        success = False
        
        if entity_type == 'emails':
            success = db_manager.delete_email(entity_id)
        elif entity_type == 'pdfs':
            success = db_manager.delete_pdf_document(entity_id)
        elif entity_type == 'projects':
            success = db_manager.delete_project(entity_id)
        else:
            return jsonify({'error': f'Unknown entity type: {entity_type}'}), 400
        
        if success:
            return jsonify({
                'message': f'{entity_type.capitalize()} deleted successfully',
                'entity_id': entity_id
            })
        else:
            return jsonify({'error': f'Failed to delete {entity_type}'}), 500
    
    except Exception as e:
        logger.error(f"Error deleting {entity_type} {entity_id}: {e}")
        return jsonify({'error': f'Failed to delete {entity_type}'}), 500

@app.route('/admin/bulk-update/<entity_type>', methods=['POST'])
@require_auth
@require_admin
def bulk_update_entities(entity_type: str):
    """Bulk update multiple entities of the same type."""
    try:
        data = request.get_json()
        if not data or 'updates' not in data:
            return jsonify({'error': 'No update data provided'}), 400
        
        updates = data['updates']
        if not isinstance(updates, list):
            return jsonify({'error': 'Updates must be a list'}), 400
        
        results = []
        for update in updates:
            entity_id = update.get('id')
            update_data = update.get('data', {})
            
            if not entity_id:
                results.append({'id': None, 'success': False, 'error': 'Missing entity ID'})
                continue
            
            try:
                success = False
                if entity_type == 'emails':
                    success = db_manager.update_email(entity_id, update_data)
                elif entity_type == 'pdfs':
                    success = db_manager.update_pdf_document(entity_id, update_data)
                elif entity_type == 'projects':
                    success = db_manager.update_project(entity_id, update_data)
                
                results.append({
                    'id': entity_id,
                    'success': success,
                    'error': None if success else 'Update failed'
                })
            
            except Exception as e:
                results.append({
                    'id': entity_id,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'entity_type': entity_type,
            'total_updates': len(updates),
            'results': results
        })
    
    except Exception as e:
        logger.error(f"Error in bulk update for {entity_type}: {e}")
        return jsonify({'error': f'Failed to perform bulk update for {entity_type}'}), 500

@app.route('/admin/system/status', methods=['GET'])
@require_auth
@require_admin
def system_status():
    """Get system status and statistics."""
    try:
        stats = {
            'database': {
                'connected': db_manager.connection is not None,
                'tables': []
            },
            'entities': {},
            'system': {
                'timestamp': datetime.utcnow().isoformat(),
                'version': '1.0.0'
            }
        }
        
        # Get table information
        if db_manager.connection:
            with db_manager.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT table_name, 
                           (SELECT COUNT(*) FROM information_schema.tables WHERE table_name = t.table_name) as exists
                    FROM (VALUES ('emails'), ('pdf_documents'), ('email_embeddings'), ('pdf_embeddings')) AS t(table_name)
                """)
                tables = cursor.fetchall()
                
                for table_name, exists in tables:
                    if exists:
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        count = cursor.fetchone()[0]
                        stats['database']['tables'].append({
                            'name': table_name,
                            'record_count': count
                        })
        
        # Get entity counts
        stats['entities'] = {
            'emails': len(db_manager.get_all_emails()) if db_manager else 0,
            'pdfs': len(db_manager.get_all_pdf_documents()) if db_manager else 0
        }
        
        return jsonify(stats)
    
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({'error': 'Failed to get system status'}), 500

@app.route('/admin/system/maintenance', methods=['POST'])
@require_auth
@require_admin
def system_maintenance():
    """Perform system maintenance tasks."""
    try:
        data = request.get_json()
        action = data.get('action')
        
        if action == 'recompute_embeddings':
            # Recompute all embeddings
            success = db_manager.recompute_all_embeddings()
            return jsonify({
                'action': action,
                'success': success,
                'message': 'Embedding recomputation completed' if success else 'Embedding recomputation failed'
            })
        
        elif action == 'cleanup_orphaned':
            # Clean up orphaned records
            success = db_manager.cleanup_orphaned_records()
            return jsonify({
                'action': action,
                'success': success,
                'message': 'Orphaned records cleanup completed' if success else 'Cleanup failed'
            })
        
        elif action == 'optimize_database':
            # Optimize database tables
            success = db_manager.optimize_tables()
            return jsonify({
                'action': action,
                'success': success,
                'message': 'Database optimization completed' if success else 'Optimization failed'
            })
        
        else:
            return jsonify({'error': f'Unknown maintenance action: {action}'}), 400
    
    except Exception as e:
        logger.error(f"Error in system maintenance: {e}")
        return jsonify({'error': 'Failed to perform maintenance action'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

def init_database():
    """Initialize database connection."""
    global db_manager
    
    try:
        config = Config()
        db_config = config.get_db_config()
        
        if not all(db_config.get(key) for key in ['host', 'database', 'user', 'password']):
            logger.error("Incomplete database configuration")
            return False
        
        db_manager = DatabaseManager(db_config)
        if not db_manager.connect():
            logger.error("Failed to connect to database")
            return False
        
        logger.info("Database connection established")
        return True
    
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False

if __name__ == '__main__':
    if init_database():
        logger.info("Starting Admin API server...")
        app.run(host='0.0.0.0', port=8000, debug=True)
    else:
        logger.error("Failed to initialize database. Exiting.")
        exit(1)

