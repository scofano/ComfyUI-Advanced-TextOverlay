"""
Font utilities for ComfyUI nodes.
Provides cross-platform font discovery and management functionality.
"""

import os
import sys
import glob
from typing import List, Dict, Optional
from pathlib import Path


class FontManager:
    """Centralized font discovery and management for ComfyUI nodes."""
    
    def __init__(self):
        self._font_cache: Optional[List[str]] = None
        self._name_to_path: Dict[str, str] = {}
        self._scanned = False
    
    def get_available_fonts(self) -> List[str]:
        """
        Returns a sorted list of available font names for dropdown menus.
        Caches results for performance.
        """
        if not self._scanned:
            self._scan_system_fonts()
        
        return sorted(self._name_to_path.keys())
    
    def get_font_path(self, font_name: str) -> str:
        """
        Converts a font name to its actual file path.
        Returns the original name if not found in our cache (backward compatibility).
        """
        if not self._scanned:
            self._scan_system_fonts()
        
        return self._name_to_path.get(font_name, font_name)
    
    def _scan_system_fonts(self) -> None:
        """Platform-specific font discovery."""
        if self._scanned:
            return
        
        self._name_to_path = {}
        
        if sys.platform == "win32":
            self._scan_windows_fonts()
        elif sys.platform == "darwin":  # macOS
            self._scan_macos_fonts()
        else:  # Linux and other Unix-like systems
            self._scan_linux_fonts()
        
        # Add some common fallback fonts
        self._add_fallback_fonts()
        
        self._scanned = True
    
    def _scan_windows_fonts(self) -> None:
        """Scan Windows system fonts directory."""
        try:
            fonts_dir = Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts"
            if fonts_dir.exists():
                # Common font file extensions
                patterns = ["*.ttf", "*.otf", "*.ttc"]
                
                for pattern in patterns:
                    for font_file in fonts_dir.glob(pattern):
                        try:
                            # Extract font name from filename (remove extension)
                            font_name = font_file.stem
                            # Use the full path as the value
                            self._name_to_path[font_name] = str(font_file)
                        except Exception:
                            continue
        except Exception:
            pass
    
    def _scan_macos_fonts(self) -> None:
        """Scan macOS system fonts directories."""
        font_dirs = [
            Path("/System/Library/Fonts"),
            Path("/Library/Fonts"),
            Path.home() / "Library/Fonts"
        ]
        
        for fonts_dir in font_dirs:
            if fonts_dir.exists():
                self._scan_directory(fonts_dir)
    
    def _scan_linux_fonts(self) -> None:
        """Scan Linux system fonts directories."""
        font_dirs = [
            Path("/usr/share/fonts"),
            Path("/usr/local/share/fonts"),
            Path.home() / ".fonts",
            Path.home() / ".local/share/fonts"
        ]
        
        for fonts_dir in font_dirs:
            if fonts_dir.exists():
                self._scan_directory(fonts_dir)
    
    def _scan_directory(self, fonts_dir: Path) -> None:
        """Scan a directory for font files."""
        patterns = ["*.ttf", "*.otf", "*.ttc", "*.woff", "*.woff2"]
        
        for pattern in patterns:
            for font_file in fonts_dir.glob(f"**/{pattern}"):
                try:
                    font_name = font_file.stem
                    # Only add if we don't already have this name (avoid duplicates)
                    if font_name not in self._name_to_path:
                        self._name_to_path[font_name] = str(font_file)
                except Exception:
                    continue
    
    def _add_fallback_fonts(self) -> None:
        """Add common fallback fonts that might not be detected."""
        fallbacks = {
            "Arial": "Arial",
            "Times New Roman": "Times New Roman", 
            "Courier New": "Courier New",
            "Helvetica": "Helvetica",
            "Verdana": "Verdana",
            "Georgia": "Georgia",
            "Trebuchet MS": "Trebuchet MS",
            "Impact": "Impact",
            "Comic Sans MS": "Comic Sans MS",
            "Arial Black": "Arial Black",
            "Tahoma": "Tahoma",
            "Lucida Sans Unicode": "Lucida Sans Unicode",
            "MS Sans Serif": "MS Sans Serif",
            "MS Serif": "MS Serif",
            "Palatino Linotype": "Palatino Linotype",
            "Arial Narrow": "Arial Narrow"
        }
        
        for name, path in fallbacks.items():
            if name not in self._name_to_path:
                self._name_to_path[name] = path


# Global instance for easy reuse across nodes
font_manager = FontManager()


def get_available_fonts() -> List[str]:
    """Convenience function to get available fonts."""
    return font_manager.get_available_fonts()


def get_font_path(font_name: str) -> str:
    """Convenience function to get font path from name."""
    return font_manager.get_font_path(font_name)