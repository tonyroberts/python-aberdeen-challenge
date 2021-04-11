from bs4 import BeautifulSoup
from bs4.element import NavigableString
from pyxll import xl_func, DataFrameFormatter, Formatter, plot
import matplotlib.pyplot as plt
import pandas as pd
import aiohttp
import asyncio
import re

_formula_one_drivers_url = "https://en.wikipedia.org/wiki/List_of_Formula_One_drivers"

_column_types = {
    "Driver_Name": str,
    "Nationality": str,
    "Seasons_Competed": str,
    "Drivers_Championships": int,
    "Race_Entries": int,
    "Race_Starts": int,
    "Pole_Positions": int,
    "Race_Wins": int,
    "Podiums": int,
    "Fastest_laps": int,
    "Points": float
}

_df_formatter = DataFrameFormatter(
    header={"interior_color": Formatter.rgb(0x20, 0x37, 0x64), "text_color": Formatter.rgb(0xFF, 0xFF, 0xFF)},
    rows=[
        {"interior_color": Formatter.rgb(0xD9, 0xE1, 0xF2)},
        {"interior_color": Formatter.rgb(0xC7, 0xD5, 0xF1)}
    ]
)


def _get_value(element, col=None):
    """Utility function to get the first value from a bs4 element
    and convert to the right type for the column (if specified).
    """
    dtype = _column_types.get(col, str)

    # If the dtype is str return the whole text for the element, removing any [] note
    if dtype is str:
        value = element.text.strip()
        match = re.match(r"^(.+)\s*\[.+\]$", value)
        if match:
            value = match.group(1)
        return value.strip("~*^")

    # Otherwise get the value from the first text element
    if isinstance(element, NavigableString):
        value = str(element).strip()

        # Convert 'num1 (num2)' to 'num1'
        if dtype in (float, int):
            match = re.match(r"(\d+(?:\.\d*)?)\s+\(\d+(?:\.\d*)?\)", value)
            if match is not None:
                value = match.group(1)

        # Convert to dtype and return
        return dtype(value)

    # Find the first child (recursively) that's a NavigableString
    for child in element:
        value = _get_value(child, col)
        if value is not None:
            return value

    return None


@xl_func(": object")  # Return the DataFrame to Excel as an object handle
async def load_f1_stats():
    """Loads a dataset of Formula One driver stats from Wikipedia."""
    async with aiohttp.ClientSession() as session:
        async with session.get(_formula_one_drivers_url) as response:
            # If we got a 429 error wait a second and try again
            if response.status == 429:
                delay = int(response.headers.get("Retry-After", "1"))
                await asyncio.sleep(delay)
                return await load_f1_stats()

            # Otherwise check the status and read the response
            assert response.status == 200, f"Request failed: {response.status}"
            data = await response.read()

    # Parse the data with BeautifulSoup
    soup = BeautifulSoup(data, "html.parser")

    # Look for the table captioned 'Formula One drivers by name'
    table = soup.find("table", {"class": "wikitable"})
    while table is not None:
        if table.find("caption").text.strip() == "Formula One drivers by name":
            break
        table = table.find_next("table", {"class": "wikitable"})

    if not table:
        raise RuntimeError("'Formula One drivers by name' table not found.")

    # Build a DataFrame from the table
    rows = table.find_all("tr")
    columns = [_get_value(h).replace(" ", "_").replace("'", "") for h in rows[0].find_all("th")]
    values = []
    for row in rows[1:]:
        data = row.find_all("td")
        if data:
            values.append([_get_value(d, c) for c, d in zip(columns, data)])

    return pd.DataFrame(values, columns=columns).infer_objects()


@xl_func("dataframe df, str[] col, bool asc: object")
def df_sort_values(df: pd.DataFrame, col, ascending=True):
    """Sort a DataFrame by a column or list of columns.
    Returns a the sorted DataFrame as an object.
    """
    return df.sort_values(col, ascending=ascending)


@xl_func("dataframe df, int n: dataframe", formatter=_df_formatter, auto_resize=True)
def df_head(df, n=None):
    """Return the first n rows of a DataFrame.
    If n is not specified the entire DataFrame is returned.
    """
    if n is None:
        return df
    return df.head(n)


@xl_func("dataframe df, str[] expressions: object")
def df_filter(df, expressions):
    """Filter a DataFrame and return a new DataFrame where all expressions
    evaluate to True.
    """
    mask = pd.Series(True, index=df.index)
    for expr in expressions:
        mask &= df.eval(expr)
    return df[mask]


@xl_func("dataframe df, str[] expressions: object")
def df_eval(df, expressions):
    """Evaluate one or more expressions and return a new DataFrame with the
    contents of the original DataFrame and the results.

    Expressions should be of the form 'C = A + B'.
    """
    expr = "\n".join(expressions)
    return df.eval(expr)


@xl_func("dataframe df, str[] groupby, str[] columns, str aggfunc")
def df_groupby(df, groupby, columns, aggfunc):
    """Group a DataFrame by a column or columns and aggregate a set
    of columns using an aggregation function (eg 'sum' or 'mean').
    """
    grouped = df.groupby(groupby)[columns]
    aggregated = getattr(grouped, aggfunc)()
    return aggregated.reset_index()


@xl_func("dataframe df, str kind, str[] columns, str index: str")
def df_plot(df, kind="line", columns=None, index=None):
    """Plot a DataFrame as an Excel shape."""
    if index is not None:
        df = df.set_index(index)

    if columns:
        df = df[columns]

    df.plot(kind=kind)
    plt.tight_layout()
    plot()

    return "[DONE]"


if __name__ == "__main__":
    # TIP: Test your code by running it as a script before trying it in Excel.
    # Think about how the Excel user will call your functions.

    # Load the data
    loop = asyncio.get_event_loop()
    f1_stats = loop.run_until_complete(load_f1_stats())

    # Top ten drivers by Race Wins
    sorted = df_sort_values(f1_stats, ["Race_Wins"], False)
    top_10 = df_head(sorted, 10)
    print("Top ten drivers by Race Wins")
    print(top_10.to_string())

    # Lowest Ratio of 'Race Entries' to 'Points' where Points > 0
    filtered = df_filter(f1_stats, ["Points > 0"])
    evaled = df_eval(filtered, ["RE_Points_Ratio = Race_Entries / Points"])
    sorted = df_sort_values(evaled, ["RE_Points_Ratio"], True)
    lowest = df_head(sorted, 10)
    print("\n\nLowest Ratio of 'Race Entries' to 'Points' where Points > 0")
    print(lowest.to_string())

    # Top 5 Nationalities by mean Race_Wins
    grouped = df_groupby(f1_stats, ["Nationality"], ["Race_Wins"], "mean")
    sorted = df_sort_values(grouped, ["Race_Wins"], False)
    top_5 = df_head(sorted, 5)
    print("\n\nTop 5 Nationalities by mean Race_Wins")
    print(top_5.to_string())
