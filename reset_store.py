#!/usr/bin/env python
"""Drop and recreate the store named by a palaestrAI runtime config.

    python reset_store.py runtime_ab_run3.conf.yaml

The database name is read from the config's ``store_uri`` rather than typed on
the command line, so this cannot target the wrong store on a machine where the
host, port or credentials differ (workstation vs laptop). The schema is NOT
created here -- run ``palaestrai -c <config> database-create`` afterwards, or
the store receiver disables itself on the first write and the whole run
completes while recording nothing.

DESTRUCTIVE: the old contents are gone. Active connections are terminated
first, since palaestrAI leaves idle sessions behind after a run and Postgres
refuses to drop a database that still has any.
"""
import sys
from urllib.parse import urlparse

import psycopg2
import yaml


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    uri = yaml.safe_load(open(argv[1]))["store_uri"]
    db = urlparse(uri).path.lstrip("/")
    if not db or db in ("postgres", "template0", "template1"):
        print(f"refusing to drop {db!r}")
        return 1

    con = psycopg2.connect(uri.rsplit("/", 1)[0] + "/postgres")
    con.autocommit = True
    cur = con.cursor()
    cur.execute(
        "select pg_terminate_backend(pid) from pg_stat_activity "
        "where datname = %s and pid <> pg_backend_pid()",
        (db,),
    )
    killed = cur.rowcount
    cur.execute(f'DROP DATABASE IF EXISTS "{db}"')
    cur.execute(f'CREATE DATABASE "{db}"')
    print(f"reset {db} (terminated {killed} stale session(s))")
    print(f"now run:  palaestrai -c {argv[1]} database-create")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
