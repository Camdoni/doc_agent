"""
watcher.py

Keeps the index fresh while docagent is open. Save a note in Obsidian and a
few seconds later the new text is searchable, with no /reindex or restart.

HOW IT WORKS:
A background thread (a second line of work running next to the chat) checks
your notes' modified times every few seconds. When something changed it waits
until the changes stop, because editors and cloud sync often write a file more
than once. Then it rebuilds the index and hands the fresh copy to the chat
in one step, so a question never sees a half-built index.
"""

import threading

from ingest import ingest, vault_changes, scan_notes


class NotesWatcher(threading.Thread):
    def __init__(self, vault, data_dir, embedder, on_update, interval=3.0):
        super().__init__(daemon=True)  # daemon: dies automatically when docagent exits
        self.vault = vault
        self.data_dir = data_dir
        self.embedder = embedder
        self.on_update = on_update   # called with a short reason like "1 edited" after each rebuild
        self.interval = interval
        self._lock = threading.Lock()  # one rebuild at a time, whether from here or from /reindex
        self._stop_event = threading.Event()

    def rebuild(self, verbose=False):
        """Rebuilds the index now. Used by the background loop and by the /reindex command."""
        with self._lock:
            return ingest(self.vault, self.data_dir, self.embedder, verbose)

    def stop(self):
        self._stop_event.set()

    def run(self):
        last_seen = None
        while not self._stop_event.wait(self.interval):
            try:
                reason = vault_changes(self.vault, self.data_dir)
                if reason is None:
                    last_seen = None
                    continue
                snapshot = scan_notes(self.vault)
                if snapshot != last_seen:
                    last_seen = snapshot  # still changing, check again next round
                    continue
                self.rebuild()
                last_seen = None
                if not self._stop_event.is_set():  # stopped mid-rebuild (the user switched folders): stay quiet
                    self.on_update(reason)
            except (Exception, SystemExit):
                last_seen = None  # e.g. the notes folder is briefly unavailable; try again next round
