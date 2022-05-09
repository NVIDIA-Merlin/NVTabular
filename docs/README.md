# Documentation

This folder contains the scripts necessary to build NVTabular's documentation.
You can view the generated [NVTabular documentation here](https://nvidia-merlin.github.io/NVTabular/main/Introduction.html).

## Contributing to Docs

Follow the instructions below to build the docs.

## Steps to follow:

1. To build the docs, you need to install a developer environment:

   ```shell
   python3 -m vevn .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   python -m pip install -r requirements-dev.txt
   ```

   > If you add or change dependencies, review the `ci/build_and_test.sh` file
   > and make a similar change to the `pip install` stanzas.

   Alternatively, you might be able use a Conda environment. See the [installation instructions](https://github.com/NVIDIA/NVTabular).

1. Build the documentation:

   ```shell
   make -C docs clean html
   ```

   This runs Sphinx in your shell and outputs to `docs/build/html/`.

1. Start an HTTP server and review your updates:

   ```shell
   python -m http.server 8000 -d docs/build/html
   ```

   Navigate a web browser to the IP address or hostname of the host machine at port 8000:

   `https://localhost:8000`

   Check that your docs edits formatted correctly, and read well.

## Decisions

### Source management: README and index files

* To preserve Sphinx's expectation that all source files are child files and directories
  of the `docs/source` directory, other content, such as the `notebooks` directory is
  copied to the source directory. You can determine which directories are copied by
  viewing `docs/source/conf.py` and looking for the `copydirs_additional_dirs` list.
  Directories are specified relative to the Sphinx source directory, `docs/source`.

* One consequence of the preceding bullet is that any change to the original files,
  such as adding or removing a topic, requires a similar change to the `docs/source/toc.yaml`
  file.  Updating the `docs/source/toc.yaml` file is not automatic.

* Because the GitHub browsing expectation is that a `README.md` file is rendered when you
  browse a directory, when a directory is copied, the `README.md` file is renamed to
  `index.md` to meet the HTML web server expectation of locating an `index.html` file
  in a directory.

* Add the file to the `docs/source/toc.yaml` file.  Keep in mind that notebooks are
  copied into the `docs/source/` directory, so the paths are relative to that location.
  Follow the pattern that is already established and you'll be fine.
  
### Adding links

TIP: When adding a link to a method or any heading that has underscores in it, repeat
the underscores in the link even though they are converted to hyphens in the HTML.

Refer to the following examples from HugeCTR:

* `../QAList.md#24-how-to-set-workspace_size_per_gpu_in_mb-and-slot_size_array`
* `./api/python_interface.md#save_params_to_files-method`

#### Docs-to-docs links

There is no concern for the GitHub browsing experience for files in the `docs/source/` directory.
You can use a relative path for the link.  For example, the following link is in the
`docs/source/hugectr_user_guide.md` file and links to the "Build HugeCTR from Source" heading
in the `docs/source/hugectr_contributor_guide.md` file:

```markdown
To build HugeCTR from scratch, refer to
[Build HugeCTR from source code](./hugectr_contributor_guide.md#build-hugectr-from-source).
```

#### Docs-to-repository links

To refer a reader to a README or program in a repository directory, state that
the link is to the repository:

```markdown
Refer to the sample Python programs in the
[examples/blah](https://github.com/NVIDIA-Merlin/NVTabular/tree/main/examples/blah)
directory of the repository.
```

The idea is to let a reader know that following the link&mdash;whether from an HTML docs page or
from browsing GitHub&mdash;results in viewing our repository on GitHub.

> TIP: In the `release_notes.md` file, use the tag such as `v3.5` instead of `master` so that
> the link is durable.

#### Links to notebooks

The notebooks are published as documentation. The few exceptions are identified in the
`docs/source/conf.py` file in the `exclude_patterns` list:

```python
exclude_patterns = [
    # list RST, MD, and IPYNB files to ignore here
]
```

If the document that you link from is also published as docs, such as `release_notes.md`, then
a relative path works both in the HTML docs page and in the repository browsing experience:

```markdown
### Some awesome feature
    + ...snip...
    + ...snip...
    + Added the [awesome notebook](examples/awesome_notebook.ipynb) to show how to use the feature.
```

#### Links from notebooks to docs

Use a link to the HTML page like the following:

```markdown
<https://nvidia-merlin.github.io/NVTabular/main/Introduction.html>
```

> I'd like to change this in the future. My preference would be to use a relative
> path, but I need to research and change how Sphinx handles relative links.
