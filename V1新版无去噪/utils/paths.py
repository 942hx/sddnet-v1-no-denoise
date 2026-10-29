"""Resolve every writable path inside this project, without following links."""
from pathlib import Path
from .runtime import PROJECT_ROOT


def output_path(value):
    candidate=Path(str(value).replace('\\','/'))
    if '..' in candidate.parts:
        raise ValueError('Output paths must not contain ..')
    candidate=candidate if candidate.is_absolute() else PROJECT_ROOT/candidate
    # Win32 aliases trailing dots/spaces and NTFS stream syntax can resolve to
    # a different object than their spelling suggests. Reject ambiguous names.
    if any(part.endswith((' ','.')) or ':' in part for part in candidate.parts[1:]):
        raise ValueError('Output paths must not use trailing-dot/space aliases or stream syntax.')
    root=PROJECT_ROOT.resolve()
    for part in [candidate,*candidate.parents]:
        if part.is_symlink() or (part.exists() and getattr(part.lstat(),'st_file_attributes',0) & 0x400):
            raise ValueError('Output paths must not traverse links/junctions.')
        if part==PROJECT_ROOT:break
    resolved=candidate.resolve()
    if not resolved.is_relative_to(root) or resolved==root:
        raise ValueError('Output must be a child directory/file inside the project.')
    if resolved.relative_to(root).parts[0].casefold() in {'data','cache','results','.git','.agents','.codex','.aws'}:
        raise ValueError('Output cannot modify data, dependencies, frozen results or project metadata.')
    return resolved


def new_output(value):
    path=output_path(value)
    if path.exists():
        raise FileExistsError('Output already exists; choose a new directory.')
    return path
