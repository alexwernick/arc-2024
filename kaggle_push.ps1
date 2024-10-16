Get-ChildItem -Path ".\kaggle_data\arc_2024\" | Remove-Item -Recurse -Force
Copy-Item -Path ".\arc_2024\data_management\" -Destination ".\kaggle_data\arc_2024\" -Recurse -Force
Copy-Item -Path ".\arc_2024\inductive_logic_programming\" -Destination ".\kaggle_data\arc_2024\" -Recurse -Force
Copy-Item -Path ".\arc_2024\representations\" -Destination ".\kaggle_data\arc_2024\" -Recurse -Force
Copy-Item -Path ".\arc_2024\runner.py" -Destination ".\kaggle_data\arc_2024\runner.py" -Force
Copy-Item -Path ".\arc_2024\solver.py" -Destination ".\kaggle_data\arc_2024\solver.py" -Force
Copy-Item -Path ".\arc_2024\grid_size_solver.py" -Destination ".\kaggle_data\arc_2024\grid_size_solver.py" -Force
Copy-Item -Path ".\arc_2024\.env.kaggle" -Destination ".\kaggle_data\arc_2024\.env" -Force
kaggle datasets version -p kaggle_data -m "Updated data" --dir-mode tar
kaggle kernels push -p arc_2024/
