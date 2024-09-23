# Copy-Item -Path "C:\Users\Alex2\AppData\Local\pypoetry\Cache\virtualenvs\arc-2024-27osIeh6-py3.12\Lib\site-packages\" `
#           -Destination ".\kaggle_packages\site_packages" `
#           -Recurse -Force `
#           -Exclude "*.zip","*.tar","*.gz","*.bz2","*.7z","*.rar","*.xz","*.tar.gz","*.tar.bz2"

poetry export -f requirements.txt --output requirements.txt --without-hashes
mkdir wheelhouse kaggle_packages/wheelhouse

# Use pip download with platform specification
pip download `
    --dest kaggle_packages/wheelhouse `
    --platform manylinux2014_x86_64 `
    --python-version 3.10 `
    --implementation cp `
    --abi cp310 `
    --only-binary=:all: `
    -r requirements.txt

#kaggle datasets version -p kaggle_packages/ -m "Updated data" --dir-mode tar
