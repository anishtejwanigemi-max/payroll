import pandas as pd
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import warnings
import numpy as np # Added for isclose check

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

class PayrollAgent:
    """
    An Agentic AI Proof of Concept for Payroll Reconciliation.
    """

    def __init__(self, gemini_api_key, file_paths):
        """
        Initializes the agent with file paths and the Gemini API key.
        :param gemini_api_key: The API key for the Gemini model.
        :param file_paths: A dictionary containing paths to the required data files.
        """
        print("Initializing Payroll Agent...")
        if not gemini_api_key:
            raise ValueError("Gemini API key is missing.")
            
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=gemini_api_key)
        self.report = []
        self.file_paths = file_paths
        self._load_data()
        print("Agent Initialized. Data loaded successfully.")

    def _load_data(self):
        """
        Data Loader Agent: Loads data from the file paths provided during initialization.
        **FULLY ROBUST**: The agent now intelligently finds and standardizes ALL critical columns
        across ALL relevant files, making it immune to naming inconsistencies.
        """
        print("Data Loader Agent: Loading all source files with universal robust settings...")

        def robust_load_file(file_path, file_desc):
            """Helper function to load either an Excel or CSV file robustly."""
            parser_config = {'encoding': 'latin1', 'engine': 'python', 'on_bad_lines': 'warn'}
            try:
                print(f"Attempting to load {file_desc} as an Excel file...")
                df = pd.read_excel(file_path)
                print(f"{file_desc} successfully loaded as Excel.")
                return df
            except Exception:
                print(f"Could not load {file_desc} as Excel, attempting to load as CSV...")
                df = pd.read_csv(file_path, **parser_config)
                print(f"{file_desc} successfully loaded as CSV.")
                return df

        try:
            self.master_data = robust_load_file(self.file_paths['master_data'], "Master Data")
            self.pay_register_may = robust_load_file(self.file_paths['pay_register_may'], "May Pay Register")
            self.pay_register_june = robust_load_file(self.file_paths['pay_register_june'], "June Pay Register")
            self.gl_data = robust_load_file(self.file_paths['gl_data'], "GL Data")
            self.sit_data = pd.read_excel(self.file_paths['sit_data'], engine='pyxlsb', sheet_name=0)
            print("All files loaded.")
        except Exception as e:
            print(f"A critical error occurred while loading data: {e}")
            raise

        # --- Data Cleaning and Standardization ---
        print("Data Loader Agent: Cleaning and standardizing all critical column names...")
        
        for df in [self.master_data, self.pay_register_may, self.pay_register_june, self.gl_data, self.sit_data]:
            df.columns = [col.strip() for col in df.columns]

        def find_and_standardize_column(df, possible_names, standard_name, file_desc):
            df_columns_lower = {col.lower(): col for col in df.columns}
            for name in possible_names:
                if name.lower() in df_columns_lower:
                    original_name = df_columns_lower[name.lower()]
                    df.rename(columns={original_name: standard_name}, inplace=True)
                    print(f"Successfully standardized '{original_name}' to '{standard_name}' in {file_desc}.")
                    return True
            raise ValueError(
                f"Could not find critical column in {file_desc}.\n\n"
                f"--> The agent looked for a column with one of these names: {possible_names}\n\n"
                f"--> However, the actual columns found are: {df.columns.tolist()}\n\n"
                "Please check the file to ensure the column name is correct, or update the agent's code."
            )

        # Define all possible names for all critical columns
        employee_id_names_master = ['SAP Emp No', 'EmployeeID']
        employee_id_names_pr = ['EE Number-Employee Details', 'EmployeeID', 'EE Number']
        employee_id_names_gl = ['Employee Number', 'EmployeeID']
        leaving_date_names = ['Leaving Date', 'Date left', 'Termination Date']
        net_pay_names = ['Net Pay-Net Payment', 'Net Pay', 'Net Payment']
        gl_paycode_names = ['EY Paycode - Description', 'Paycode Description', 'Description']

        # Standardize all critical columns
        find_and_standardize_column(self.master_data, employee_id_names_master, 'EmployeeID', "Master Data")
        find_and_standardize_column(self.pay_register_may, employee_id_names_pr, 'EmployeeID', "May Pay Register")
        find_and_standardize_column(self.pay_register_june, employee_id_names_pr, 'EmployeeID', "June Pay Register")
        find_and_standardize_column(self.gl_data, employee_id_names_gl, 'EmployeeID', "GL Data")
        find_and_standardize_column(self.master_data, leaving_date_names, 'Leaving Date', "Master Data")
        find_and_standardize_column(self.pay_register_may, net_pay_names, 'Net Pay-Net Payment', "May Pay Register")
        find_and_standardize_column(self.pay_register_june, net_pay_names, 'Net Pay-Net Payment', "June Pay Register")
        find_and_standardize_column(self.gl_data, gl_paycode_names, 'EY Paycode - Description', "GL Data")

        self.master_data['Leaving Date'] = self.master_data['Leaving Date'].astype(str)
        print("All critical columns cleaned and standardized.")


    def _generate_fallout_report(self, title, finding, analysis_steps, recommendation, data_df=None):
        """
        Exception Handler Agent: Uses the LLM to generate a report for a specific fallout.
        """
        print(f"Exception Handler Agent: Generating report for '{title}'...")
        template = """
        You are an expert Payroll Analyst Agent. Based on the provided data, generate a clear, concise fallout report.
        Follow this structure exactly:
        
        **1. Task Agent Finding:**
        {finding}

        **2. Validation Agent Analysis:**
        {analysis_steps}
        Based on the data, I have identified the following specific records:
        {data}

        **3. Recommendation:**
        {recommendation}
        """
        prompt = PromptTemplate(template=template, input_variables=["finding", "analysis_steps", "data", "recommendation"])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        data_str = data_df.to_string(index=False) if data_df is not None and not data_df.empty else "No specific records found."
        response = chain.run({'finding': finding, 'analysis_steps': analysis_steps, 'data': data_str, 'recommendation': recommendation})
        self.report.append(f"## Fallout: {title}\n\n{response}\n\n---\n")
        print("Report generated.")

    def check_missing_active_employees(self):
        active_employees = self.master_data[self.master_data['Leaving Date'].str.contains('9999', na=False)]
        missing_employees = active_employees[~active_employees['EmployeeID'].isin(self.pay_register_june['EmployeeID'])]
        if not missing_employees.empty:
            finding = "I have identified employees who are listed as 'Active' in the master data file but are missing from the current draft pay register."
            analysis = "1. Filtered the UK Master Data for active employees (Leaving Date contains '9999').\n2. Compared this active list against the employee IDs in the June Pay Register.\n3. Isolated the employees present in the master data but absent from the pay register."
            reco = "These employees are likely eligible for pay but were missed during payroll processing. It is recommended to instruct the payroll vendor to **add these employees to the pay run** to ensure they are paid correctly."
            self._generate_fallout_report("Active Employee(s) Missing from Pay Register", finding, analysis, reco, missing_employees[['EmployeeID', 'Last Name', 'First Name']])

    def check_terminated_employees_in_payroll(self):
        terminated_employees = self.master_data[~self.master_data['Leaving Date'].str.contains('9999', na=False)]
        paid_terminated = terminated_employees[terminated_employees['EmployeeID'].isin(self.pay_register_june['EmployeeID'])]
        if not paid_terminated.empty:
            report_df = pd.merge(paid_terminated, self.pay_register_june[['EmployeeID', 'Net Pay-Net Payment']], on='EmployeeID')
            finding = "I have identified employees who are listed as 'Separated' in the master data file but are still included in the current draft pay register."
            analysis = "1. Filtered the UK Master Data for terminated employees (Leaving Date does not contain '9999').\n2. Checked if any of these employee IDs exist in the June Pay Register.\n3. Confirmed that these terminated employees are scheduled to receive a payment."
            reco = "These employees are not eligible for pay in the current cycle. To prevent overpayment, it is recommended to instruct the payroll vendor to **remove these employees from the June pay run** immediately."
            self._generate_fallout_report("Terminated Employee(s) Included in Pay Register", finding, analysis, reco, report_df[['EmployeeID', 'Last Name', 'First Name', 'Leaving Date', 'Net Pay-Net Payment']])

    def check_net_pay_variance(self, threshold=0.15):
        may_pay = self.pay_register_may[['EmployeeID', 'Net Pay-Net Payment']].rename(columns={'Net Pay-Net Payment': 'MayNetPay'})
        june_pay = self.pay_register_june[['EmployeeID', 'Net Pay-Net Payment']].rename(columns={'Net Pay-Net Payment': 'JuneNetPay'})
        merged_pay = pd.merge(may_pay, june_pay, on='EmployeeID')
        # Ensure pay columns are numeric before calculation
        merged_pay['MayNetPay'] = pd.to_numeric(merged_pay['MayNetPay'], errors='coerce').fillna(0)
        merged_pay['JuneNetPay'] = pd.to_numeric(merged_pay['JuneNetPay'], errors='coerce').fillna(0)
        merged_pay = merged_pay[merged_pay['MayNetPay'] != 0]
        merged_pay['Variance'] = (merged_pay['JuneNetPay'] - merged_pay['MayNetPay']) / merged_pay['MayNetPay']
        variance_fallouts = merged_pay[abs(merged_pay['Variance']) > threshold].copy()
        variance_fallouts['Variance'] = pd.to_numeric(variance_fallouts['Variance']).map('{:.2%}'.format)
        if not variance_fallouts.empty:
            finding = f"During the 'Sense Check,' I identified employees whose Net Pay in June changed by more than {threshold:.0%} compared to May."
            analysis = f"1. Compared the 'Net Pay' column in the June Pay Register against the May Pay Register for all common employees.\n2. Calculated the percentage change in net pay for each employee.\n3. Flagged all employees where the absolute variance exceeded the {threshold:.0%} threshold."
            reco = "Large variances can be valid (e.g., due to bonuses, unpaid leave, or salary changes) but can also indicate errors. A **manual review of these employees' inputs** in the SIT file is advised to confirm the changes are legitimate."
            self._generate_fallout_report(f"Net Pay Variance > {threshold:.0%}", finding, analysis, reco, variance_fallouts[['EmployeeID', 'MayNetPay', 'JuneNetPay', 'Variance']])

    def check_gl_mismatch(self):
        gl_net_pay_filtered = self.gl_data[self.gl_data['EY Paycode - Description'] == 'Net pay'].copy()
        gl_net_pay_filtered['Total'] = pd.to_numeric(gl_net_pay_filtered['Total'], errors='coerce')
        gl_net_pay_filtered.dropna(subset=['Total'], inplace=True)
        gl_net_pay = gl_net_pay_filtered[['EmployeeID', 'Total']]
        gl_net_pay.rename(columns={'Total': 'GL_NetPay'}, inplace=True)
        gl_net_pay['GL_NetPay'] = gl_net_pay['GL_NetPay'].abs()

        pr_net_pay = self.pay_register_june[['EmployeeID', 'Net Pay-Net Payment']].copy()
        pr_net_pay['Net Pay-Net Payment'] = pd.to_numeric(pr_net_pay['Net Pay-Net Payment'], errors='coerce')
        pr_net_pay.dropna(subset=['Net Pay-Net Payment'], inplace=True)
        pr_net_pay.rename(columns={'Net Pay-Net Payment': 'PR_NetPay'}, inplace=True)

        comparison_df = pd.merge(pr_net_pay, gl_net_pay, on='EmployeeID', how='inner')
        mismatch = comparison_df[~np.isclose(comparison_df['PR_NetPay'], comparison_df['GL_NetPay'])]
        if not mismatch.empty:
            finding = "During GL validation, I found a variance between the Net Pay recorded in the Pay Register and the amount in the General Ledger file."
            analysis = "1. Extracted 'Net pay' credit entries from the GL file for each employee.\n2. Compared these values against the 'Net Pay' in the June Pay Register.\n3. Identified employees where the two amounts do not reconcile."
            reco = "The Pay Register is typically the source of truth for employee payments. The GL file appears to have incorrect entries. It is recommended to **prepare a correction journal to update the GL** to match the Pay Register values."
            self._generate_fallout_report("GL vs. Pay Register Net Pay Mismatch", finding, analysis, reco, mismatch)
    
    def run_all_checks(self):
        """
        Main orchestrator that runs all checks and returns the compiled report.
        """
        print("\nPlan Executor Agent: Starting all payroll checks...")
        self.check_missing_active_employees()
        self.check_terminated_employees_in_payroll()
        self.check_net_pay_variance()
        self.check_gl_mismatch()
        print("All checks completed.\n")
        
        if not self.report:
            self.report.append("## All Checks Passed\n\n**No discrepancies were found in the payroll data.** All validations completed successfully.")

        final_report = "# Payroll Reconciliation Agent Report - June 2025\n\n"
        final_report += "This report summarizes the findings from the automated quality checks. Each section details a specific fallout, the AI's analysis, and a recommended course of action for the Human-in-the-Loop (HITL) to review.\n\n---\n\n"
        final_report += "\n".join(self.report)
        
        print("Audit and Reporting Agent: Final report generated.")
        return final_report

