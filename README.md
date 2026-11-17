# team-B1-recsys

ML-based movie recommendation system built on the MovieLens 1M dataset.

## Data

The raw data is **not stored in Git**. It lives in a shared DagsHub storage and is
versioned with [DVC](https://dvc.org). Git only keeps a small pointer file
(`data/raw/ml-1m.dvc`) that says which version of the data goes with each commit.

| | |
|---|---|
| Shared storage | DagsHub repository [`e26-kairi/team-B1-recsys`](https://dagshub.com/e26-kairi/team-B1-recsys) |
| DVC remote | `origin` → `https://dagshub.com/e26-kairi/team-B1-recsys.dvc` |
| Local folder | `data/raw/ml-1m/` (git-ignored) |
| Cost | Free plan, no credit card linked: there is no way to be charged |

### 1. Get access (once)

1. Sign in to [dagshub.com](https://dagshub.com) with your GitHub account.
2. Ask the owner of the DagsHub repository (`e26-kairi`) to add you as a
   collaborator with **Write** permission, and accept the invitation you receive.
3. Create your personal token: DagsHub → your avatar → **Your settings** → **Tokens**.
   Copy it. **Never share it, paste it in a chat or commit it.**

### 2. Install DVC

From the root of the repository (Python 3.10 or newer):

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install "dvc>=3,<4"
```

### 3. Save your credentials (once per clone)

```bash
dvc remote modify origin --local user <your-dagshub-username>
dvc remote modify origin --local password <your-dagshub-token>
```

The `--local` flag is important: it stores the credentials in `.dvc/config.local`,
which is git-ignored. Without `--local` they would go into `.dvc/config` and end up
in Git.

### 4. Download the data

```bash
dvc pull
```

That's the only command needed. To check that you have exactly the same data as the
rest of the team:

```bash
dvc status        # must print: Data and pipelines are up to date.
```

DVC verifies every file against the hashes in `data/raw/ml-1m.dvc`. You can also
compare them by hand (`md5sum data/raw/ml-1m/*`, or `Get-FileHash -Algorithm MD5`
in PowerShell):

| File | MD5 |
|---|---|
| `movies.dat` | `ef9fd9ff8d1b33faf73e2122c371e910` |
| `ratings.dat` | `a89aa3591bc97d6d4e0c89459ff39362` |
| `users.dat` | `681c0711d0d4548cd3d4718a49c2dcd8` |
| `README` | `6a3f626147ec67b732a69beccae0aaa8` |

### What is in `data/raw/ml-1m/`

MovieLens 1M as published by [GroupLens](https://grouplens.org/datasets/movielens/1m/).
Columns are separated by `::`, files have no header, and they must be read with
`encoding="latin-1"` (some titles have accents).

| File | Rows | Columns |
|---|---|---|
| `ratings.dat` | 1,000,209 | `user_id::movie_id::rating::timestamp` (rating 1–5, Unix timestamp) |
| `movies.dat` | 3,883 | `movie_id::title::genres` (title includes the year, genres separated by `\|`) |
| `users.dat` | 6,040 | `user_id::gender::age::occupation::zip_code` (age and occupation are codes) |
| `README` | — | Original GroupLens description, licence and code tables |

### Save a new version of the data

```bash
dvc add data/raw/ml-1m            # updates the pointer data/raw/ml-1m.dvc
git add data/raw/ml-1m.dvc
git commit -m "Update raw data: <what changed>"
dvc push                          # uploads the new files to DagsHub
git push
```

Always run both `dvc push` and `git push`: one uploads the data, the other the pointer.

### Recover a previous version

Every version stays in the shared storage. To go back to the data of an older commit:

```bash
git log --oneline -- data/raw/ml-1m.dvc         # list the data versions
git checkout <commit> -- data/raw/ml-1m.dvc      # take the pointer of that version
dvc checkout                                     # or dvc pull if you don't have it locally
```

To make that version the current one again, commit the restored `data/raw/ml-1m.dvc`.

### Rules

- Never commit data files: everything inside `data/` is git-ignored except the
  `*.dvc` pointers and the `.gitignore` files that DVC creates.
- Never commit passwords or tokens: they only go in `.dvc/config.local`.

### Troubleshooting

| Problem | Fix |
|---|---|
| `401 Unauthorized` | Wrong user or token. Repeat step 3. |
| `403 Forbidden` / repository not found | You are not a collaborator yet, or you haven't accepted the invitation. |
| `dvc: command not found` | Activate the virtual environment (step 2). |
| `dvc status` shows changes | Someone changed the local files. Run `dvc checkout` to restore them. |
