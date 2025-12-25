import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.config import settings
from app.core.database import Base
from app.models.domain import *

target_metadata = Base.metadata