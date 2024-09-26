@echo on

:check
IF not EXIST "venv" (
	goto init
)
goto end

:init
python -m venv venv
call venv\Scripts\activate
call pip install -r requirements.txt
call deactivate

:end