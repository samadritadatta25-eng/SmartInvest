# ==================================================
# SMARTINVEST - DATA PROCESSING
# Reading, cleaning, validating and aligning uploaded price files.
# ==================================================

import pandas as pd

MIN_COMMON_DAYS = 252    # About one year of matching trading days

# Common names that a date column may have
DATE_NAMES = ["date", "datetime", "timestamp", "trading date", "time"]


# --------------------------------
# 1. FILE READING FUNCTIONS
# --------------------------------

def get_file_type(file):
    """Return 'csv' or 'excel' based on the file name."""

    file_name = file.name.lower()

    if file_name.endswith(".csv"):
        return "csv"
    elif file_name.endswith(".xlsx"):
        return "excel"
    else:
        raise ValueError(
            "Unsupported file format. Please upload a CSV or XLSX file."
        )


def standardize_date_column(data):
    """Rename the date column to 'Date' (e.g. 'date', ' DATE ', 'Datetime')."""

    # Remove extra spaces from column names
    data.columns = data.columns.astype(str).str.strip()

    for column in data.columns:
        if column.lower() in DATE_NAMES:
            data = data.rename(columns={column: "Date"})
            return data

    return data


def is_yahoo_layout(raw_sheet):
    """
    Yahoo Finance sheets start like this:
        Row 1: Price  | Adj Close | Close | ...
        Row 2: Ticker | DABUR.NS  | ...
    """

    if len(raw_sheet) < 3:
        return False

    first_cell = str(raw_sheet.iloc[0, 0]).strip().lower()
    second_cell = str(raw_sheet.iloc[1, 0]).strip().lower()

    return first_cell == "price" and second_cell == "ticker"


def read_yahoo_sheet(raw_sheet, sheet_name):
    """Extract Date and Adj Close from one Yahoo Finance sheet."""

    headers = [str(value).strip() for value in raw_sheet.iloc[0]]

    if "Adj Close" in headers:
        price_position = headers.index("Adj Close")
    else:
        raise ValueError(
            f"Sheet '{sheet_name}' has no 'Adj Close' column."
        )

    # Stock name = sheet name. If the sheet has a default name
    # such as 'Sheet1', use the ticker (e.g. ^NSEI) instead.
    column_name = sheet_name
    if sheet_name.lower().startswith("sheet"):
        column_name = str(raw_sheet.iloc[1, price_position]).strip()

    # Header rows contain text, so they become invalid dates
    # and are removed later in the cleaning step.
    sheet_data = pd.DataFrame({
        "Date": raw_sheet.iloc[1:, 0].values,
        column_name: raw_sheet.iloc[1:, price_position].values
    })

    sheet_data["Date"] = pd.to_datetime(
        sheet_data["Date"], errors="coerce"
    )
    sheet_data = sheet_data.dropna(subset=["Date"])

    return sheet_data


def read_price_file(file):
    """
    Read an uploaded price file and return a table:
        Date | Price column 1 | Price column 2 | ...

    Supported formats:
    1. Yahoo Finance Excel workbook (one sheet per security)
    2. Ordinary CSV or Excel table with a Date column
    """

    file_type = get_file_type(file)
    notes = []

    # ---- CSV file ----
    if file_type == "csv":
        data = pd.read_csv(file)
        data = standardize_date_column(data)
        return data, notes

    # ---- Excel file: read every sheet without headers ----
    all_sheets = pd.read_excel(file, sheet_name=None, header=None)

    first_sheet_name = list(all_sheets.keys())[0]
    first_sheet = all_sheets[first_sheet_name]

    # Yahoo Finance workbook: combine all sheets by date
    if is_yahoo_layout(first_sheet):

        combined_data = None

        for sheet_name, raw_sheet in all_sheets.items():

            if not is_yahoo_layout(raw_sheet):
                notes.append(
                    f"Sheet '{sheet_name}' skipped: not in Yahoo Finance format."
                )
                continue
            sheet_data = read_yahoo_sheet(raw_sheet, sheet_name)

            if combined_data is None:
                combined_data = sheet_data
            else:
                # Outer merge keeps every date for now.
                # Dates missing in any sheet are handled in cleaning.
                combined_data = combined_data.merge(
                    sheet_data, on="Date", how="outer"
                )

        notes.append(
            "Yahoo Finance format detected: 'Adj Close' prices used."
        )
        return combined_data, notes

    # Ordinary Excel table: first row is the header
    data = first_sheet.copy()
    data.columns = data.iloc[0]
    data = data.iloc[1:].reset_index(drop=True)
    data = standardize_date_column(data)

    if len(all_sheets) > 1:
        notes.append(
            f"Only the first sheet ('{first_sheet_name}') was used."
        )

    return data, notes


# --------------------------------
# 2. DATA CLEANING FUNCTION
# --------------------------------

def clean_price_data(data, file_label):
    """
    Clean one price table and return:
        cleaned table, list of price columns, cleaning report
    """

    if data is None or "Date" not in data.columns:
        raise ValueError(
            f"{file_label}: no date column found. "
            f"Columns detected: {list(data.columns) if data is not None else []}. "
            "Please name the date column 'Date'."
        )

    report = {"File": file_label, "Rows read": len(data)}

    # Invalid dates
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    report["Invalid dates removed"] = int(data["Date"].isna().sum())
    data = data.dropna(subset=["Date"])

    # Duplicate dates (keep the first occurrence)
    report["Duplicate dates removed"] = int(data["Date"].duplicated().sum())
    data = data.drop_duplicates(subset="Date", keep="first")

    # Sort by date
    data = data.sort_values("Date").reset_index(drop=True)

    # Identify price columns
    price_columns = [col for col in data.columns if col != "Date"]

    if len(price_columns) == 0:
        raise ValueError(f"{file_label}: no price columns found.")

    # Convert prices to numbers (text becomes missing)
    for col in price_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Zero or negative prices are invalid -> treat as missing
    nonpositive_count = int((data[price_columns] <= 0).sum().sum())
    data[price_columns] = data[price_columns].where(data[price_columns] > 0)
    report["Non-positive prices removed"] = nonpositive_count

    # Unusable columns: no valid prices at all
    valid_counts = data[price_columns].count()
    unusable = valid_counts[valid_counts == 0].index.tolist()

    if len(unusable) > 0:
        raise ValueError(
            f"{file_label}: these columns have no valid prices: "
            + ", ".join(str(name) for name in unusable)
        )

    # Missing prices: we do NOT invent prices.
    # Dates with any missing price are removed.
    report["Missing prices"] = int(data[price_columns].isna().sum().sum())
    rows_before = len(data)
    data = data.dropna(subset=price_columns).reset_index(drop=True)
    report["Incomplete dates removed"] = rows_before - len(data)

    report["Clean rows"] = len(data)

    return data, price_columns, report


# --------------------------------
# 3. ALIGNMENT FUNCTION
# --------------------------------

def align_stock_and_market(stock_prices, market_data, market_column):
    """Keep only dates that exist in BOTH files (no row-position matching)."""

    aligned_data = stock_prices.merge(market_data, on="Date", how="inner")
    aligned_data = aligned_data.sort_values("Date").reset_index(drop=True)

    if len(aligned_data) < MIN_COMMON_DAYS:
        raise ValueError(
            f"Only {len(aligned_data)} matching trading days were found "
            f"between the stock file and the benchmark file. "
            f"At least {MIN_COMMON_DAYS} (about one year) are required "
            "for reliable beta and alpha estimates."
        )

    return aligned_data


# --------------------------------
# 4. DAILY RETURNS
# --------------------------------

def calculate_daily_returns(aligned_data):
    """Daily return = (Today's price / Yesterday's price) - 1"""

    prices = aligned_data.set_index("Date")
    daily_returns = prices.pct_change()

    # The first day has no previous price, so it is removed
    daily_returns = daily_returns.dropna()

    return daily_returns
