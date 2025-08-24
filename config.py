#!/usr/bin/env python3
"""
Configuration management for Email Chunker

Loads database credentials from environment variables or local config file.
The config file is gitignored for security.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for database and application settings."""
    
    def __init__(self, config_file: str = "config.json"):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to configuration file (default: config.json)
        """
        self.config_file = Path(config_file)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or environment variables."""
        config = {}
        
        # Try to load from config file first
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                print(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
        
        # Override with environment variables (higher priority)
        config.update(self._load_from_env())
        
        return config
    
    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}
        
        # Database configuration
        env_mapping = {
            'DB_HOST': 'db_host',
            'DB_PORT': 'db_port',
            'DB_NAME': 'db_name',
            'DB_USER': 'db_user',
            'DB_PASSWORD': 'db_password',
            'MAILDIR_PATH': 'maildir_path',
            'EMBEDDING_MODEL': 'embedding_model'
        }
        
        for env_var, config_key in env_mapping.items():
            value = os.getenv(env_var)
            if value:
                # Convert port to integer
                if env_var == 'DB_PORT':
                    try:
                        env_config[config_key] = int(value)
                    except ValueError:
                        print(f"Warning: Invalid DB_PORT value: {value}")
                else:
                    env_config[config_key] = value
        
        return env_config
    
    def get_db_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return {
            'host': self.config.get('db_host'),
            'port': self.config.get('db_port', 5432),
            'database': self.config.get('db_name'),
            'user': self.config.get('db_user'),
            'password': self.config.get('db_password')
        }
    
    def get_maildir_path(self) -> Optional[str]:
        """Get Maildir path from configuration."""
        return self.config.get('maildir_path')
    
    def get_embedding_model(self) -> str:
        """Get embedding model name from configuration."""
        return self.config.get('embedding_model', 'intfloat/e5-base')
    
    def has_db_config(self) -> bool:
        """Check if database configuration is complete."""
        db_config = self.get_db_config()
        required = ['host', 'database', 'user', 'password']
        return all(db_config.get(key) for key in required)
    
    def create_config_file(self, db_host: str, db_port: int, db_name: str, 
                          db_user: str, db_password: str, maildir_path: str = None,
                          embedding_model: str = 'intfloat/e5-base') -> bool:
        """
        Create a configuration file with the provided settings.
        
        Args:
            db_host: Database host
            db_port: Database port
            db_name: Database name
            db_user: Database user
            db_password: Database password
            maildir_path: Optional Maildir path
            embedding_model: Embedding model name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config_data = {
                'db_host': db_host,
                'db_port': db_port,
                'db_name': db_name,
                'db_user': db_user,
                'db_password': db_password,
                'maildir_path': maildir_path,
                'embedding_model': embedding_model
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            print(f"Configuration saved to {self.config_file}")
            print("Note: This file is gitignored for security")
            
            # Reload configuration
            self.config = self._load_config()
            return True
            
        except Exception as e:
            print(f"Error creating config file: {e}")
            return False
    
    def show_config(self, show_password: bool = False):
        """Display current configuration (optionally hiding password)."""
        print("Current Configuration:")
        print("=" * 40)
        
        db_config = self.get_db_config()
        for key, value in db_config.items():
            if key == 'password' and not show_password:
                print(f"  {key}: {'*' * 8}")
            else:
                print(f"  {key}: {value}")
        
        print(f"  maildir_path: {self.get_maildir_path()}")
        print(f"  embedding_model: {self.get_embedding_model()}")
        print()
        
        if self.has_db_config():
            print("✅ Database configuration is complete")
        else:
            print("❌ Database configuration is incomplete")
            missing = []
            db_config = self.get_db_config()
            for key in ['host', 'database', 'user', 'password']:
                if not db_config.get(key):
                    missing.append(key)
            print(f"   Missing: {', '.join(missing)}")


def create_config_interactive() -> bool:
    """Interactive configuration setup."""
    print("Email Chunker Configuration Setup")
    print("=" * 40)
    print()
    
    # Get database settings
    db_host = input("Database host (default: localhost): ").strip() or "localhost"
    db_port_input = input("Database port (default: 5432): ").strip() or "5432"
    try:
        db_port = int(db_port_input)
    except ValueError:
        print("Invalid port number, using 5432")
        db_port = 5432
    
    db_name = input("Database name: ").strip()
    if not db_name:
        print("Database name is required")
        return False
    
    db_user = input("Database user: ").strip()
    if not db_user:
        print("Database user is required")
        return False
    
    db_password = input("Database password: ").strip()
    if not db_password:
        print("Database password is required")
        return False
    
    # Optional settings
    maildir_path = input("Maildir path (optional): ").strip() or None
    embedding_model = input("Embedding model (default: intfloat/e5-base): ").strip() or "intfloat/e5-base"
    
    # Create configuration
    config = Config()
    success = config.create_config_file(
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
        db_user=db_user,
        db_password=db_password,
        maildir_path=maildir_path,
        embedding_model=embedding_model
    )
    
    if success:
        print("\nConfiguration setup complete!")
        config.show_config()
        return True
    else:
        print("\nConfiguration setup failed!")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        create_config_interactive()
    elif len(sys.argv) > 1 and sys.argv[1] == "show":
        config = Config()
        config.show_config()
    elif len(sys.argv) > 1 and sys.argv[1] == "show-full":
        config = Config()
        config.show_config(show_password=True)
    else:
        print("Usage:")
        print("  python config.py setup    - Interactive configuration setup")
        print("  python config.py show     - Show current configuration")
        print("  python config.py show-full - Show full configuration (including password)")
        print()
        print("Or set environment variables:")
        print("  export DB_HOST=localhost")
        print("  export DB_PORT=5432")
        print("  export DB_NAME=email_archive")
        print("  export DB_USER=username")
        print("  export DB_PASSWORD=password")
