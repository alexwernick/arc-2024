Get-ChildItem -Path ".\kaggle_data\arc_2024\" | Remove-Item -Recurse -Force
Copy-Item -Path ".\arc_2024\data_management\" -Destination ".\kaggle_data\arc_2024\" -Recurse -Force
Copy-Item -Path ".\arc_2024\.env.kaggle" -Destination ".\kaggle_data\arc_2024\.env" -Force
#Copy-Item -Path ".\arc_2024\__init__.py" -Destination ".\kaggle_data\arc_2024\__init__.py" -Force
kaggle datasets version -p kaggle_data/ -m "Updated data" --dir-mode tar
kaggle kernels push -p arc_2024/
