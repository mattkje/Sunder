"""Just enough of the Jellyfin API to browse the music library and find each track's file on disk.

Configured with environment variables:
- JELLYFIN_URL: e.g. http://jellyfin:8096
- JELLYFIN_API_KEY: an API key from Jellyfin's Dashboard -> API Keys (it can read file paths)
- JELLYFIN_PATH_MAP: optional "<jellyfin prefix>=<local prefix>" pairs, comma-separated, when this service
  sees the music folder at a different path than Jellyfin does (e.g. "/media/music=/music").
"""

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

URL = os.environ.get("JELLYFIN_URL", "").rstrip("/")
API_KEY = os.environ.get("JELLYFIN_API_KEY", "")
PATH_MAP = [tuple(pair.split("=", 1)) for pair in os.environ.get("JELLYFIN_PATH_MAP", "").split(",") if "=" in pair]

# Generated files are named "<original name> [Instrumental].flac" / "[Vocals].flac" and titled the same way.
# Square brackets rather than parentheses: libraries already contain real tracks called "... (Instrumental)",
# and Coral hides anything carrying these exact suffixes.
STEMS = ("Instrumental", "Vocals")


def configured() -> bool:
    return bool(URL and API_KEY)


def is_stem(title: str) -> bool:
    return any(title.endswith(f" [{stem}]") for stem in STEMS)


def albums(search: str = "", start: int = 0, limit: int = 60) -> dict:
    query = {"IncludeItemTypes": "MusicAlbum", "Recursive": "true", "SortBy": "AlbumArtist,SortName",
             "StartIndex": start, "Limit": limit, "Fields": "ChildCount", "ImageTypeLimit": 1}
    if search:
        query["searchTerm"] = search
    result = _get("/Items", query)
    return {"total": result.get("TotalRecordCount", 0),
            "albums": [{"id": a["Id"], "name": a.get("Name", ""), "artist": a.get("AlbumArtist", ""),
                        "year": a.get("ProductionYear"), "tracks": a.get("ChildCount"),
                        "image": bool(a.get("ImageTags", {}).get("Primary"))} for a in result.get("Items", [])]}


def tracks(album_id: str) -> list[dict]:
    """The album's original tracks, each with its local file and whether its stems already exist."""
    result = _get("/Items", {"ParentId": album_id, "IncludeItemTypes": "Audio", "Recursive": "true",
                             "SortBy": "ParentIndexNumber,IndexNumber,SortName", "Fields": "Path"})
    out = []
    for item in result.get("Items", []):
        title = item.get("Name", "")
        if is_stem(title):
            continue
        path = local_path(item.get("Path", ""))
        out.append({"id": item["Id"], "name": title, "index": item.get("IndexNumber"), "albumId": album_id,
                    "path": str(path) if path else None, "found": bool(path and path.exists()),
                    "hasStems": bool(path and all(stem_path(path, s).exists() for s in STEMS))})
    return out


def image(album_id: str, size: int = 160) -> bytes:
    request = urllib.request.Request(f"{URL}/Items/{album_id}/Images/Primary?fillWidth={size}&fillHeight={size}&quality=85",
                                     headers=_headers())
    with urllib.request.urlopen(request, timeout=15) as response:
        return response.read()


def refresh(item_id: str):
    """Asks Jellyfin to rescan an album folder so it picks up newly written files."""
    query = urllib.parse.urlencode({"Recursive": "true", "MetadataRefreshMode": "Default",
                                    "ImageRefreshMode": "Default", "ReplaceAllMetadata": "false"})
    request = urllib.request.Request(f"{URL}/Items/{item_id}/Refresh?{query}", method="POST", headers=_headers())
    urllib.request.urlopen(request, timeout=30).close()


def local_path(jellyfin_path: str) -> Path | None:
    if not jellyfin_path:
        return None
    for remote, local in PATH_MAP:
        if jellyfin_path.startswith(remote):
            return Path(local + jellyfin_path[len(remote):])
    return Path(jellyfin_path)


def stem_path(original: Path, stem: str) -> Path:
    return original.with_name(f"{original.stem} [{stem}].flac")


def _headers() -> dict:
    return {"Authorization": f'MediaBrowser Client="Sunder Server", Device="Server", DeviceId="sunder-server", '
                             f'Version="1.0", Token="{API_KEY}"'}


def _get(path: str, query: dict) -> dict:
    request = urllib.request.Request(f"{URL}{path}?{urllib.parse.urlencode(query)}", headers=_headers())
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())
