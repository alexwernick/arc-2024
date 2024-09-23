Get-ChildItem -Path ".\kaggle_data\arc_2024\" | Remove-Item -Recurse -Force
Copy-Item -Path ".\arc_2024\data_management\" -Destination ".\kaggle_data\arc_2024\data_management" -Recurse -Force
Copy-Item -Path ".\arc_2024\popper\" -Destination ".\kaggle_data\arc_2024\popper" -Recurse -Force
Copy-Item -Path ".\arc_2024\.env.kaggle" -Destination ".\kaggle_data\arc_2024\.env" -Force
Copy-Item -Path ".\arc_2024\data\arc_parsed_popper\253bf280\" -Destination ".\kaggle_data\arc_2024\data\arc_parsed_popper\253bf280" -Recurse -Force

kaggle datasets version -p kaggle_data/ -m "Updated data" --dir-mode zip
#kaggle kernels push -p arc_2024/
