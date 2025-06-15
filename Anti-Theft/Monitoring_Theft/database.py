import os
import logging
from sqlalchemy import create_engine, Column, Integer, Text, Float
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get DATABASE_URL from .env
DATABASE_URL = os.getenv("DATABASE_URL")
logger.debug(f"DATABASE_URL: {DATABASE_URL}")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL must be set in environment variables")

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define the table
class MyTable(Base):
    __tablename__ = 'anti_theft'
    id = Column(Integer, primary_key=True, autoincrement=True)
    file_name = Column(Text)
    face_image = Column(Text)         # Base64-encoded image stored as TEXT
    embedding = Column(ARRAY(Float))  # Numeric array (float)
    create_date = Column(Text)        # Store date as TEXT
    detection_time = Column(Text)
    status = Column(Text)             # Store status as TEXT

# Create the table in the database
Base.metadata.create_all(bind=engine)

print("✅ Table 'anti_theft' created successfully.")

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
