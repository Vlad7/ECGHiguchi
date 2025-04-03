

import sys
import os

# Получаем путь к родительскому каталогу
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Добавляем его в sys.path
sys.path.insert(0, parent_dir)



# Теперь можно импортировать модуль
import ECGHiguchi as EH # Замените на нужный модуль

print(EH.get_sex_for_each_id(['0001', '0002'], True))