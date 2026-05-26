import os
import sys
import sqlite3
import hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DB_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "warehouse.db")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class ETLProcessor:
    """Компонент ETL: extract -> transform -> load."""

    def __init__(self, data_dir, db_path):
        self.data_dir = data_dir
        self.db_path  = db_path

    def extract(self):
        required = ["sales.csv", "products.csv", "stores.csv"]
        missing  = [f for f in required
                    if not os.path.exists(os.path.join(self.data_dir, f))]
        if missing:
            raise FileNotFoundError(
                "Отсутствуют файлы: " + ", ".join(missing) +
                "\nОжидаются в: " + self.data_dir)
        raw = {
            "sales":    pd.read_csv(os.path.join(self.data_dir, "sales.csv")),
            "products": pd.read_csv(os.path.join(self.data_dir, "products.csv")),
            "stores":   pd.read_csv(os.path.join(self.data_dir, "stores.csv")),
        }
        print("[ETL] Извлечено: продажи={}, товары={}, магазины={}".format(
            len(raw["sales"]), len(raw["products"]), len(raw["stores"])))
        return raw

    def transform(self, raw):
        sales    = raw["sales"].copy()
        products = raw["products"].copy()
        stores   = raw["stores"].copy()

        before = len(sales)
        sales.dropna(subset=["date","product_id","store_id","quantity","price"],
                     inplace=True)
        if before - len(sales):
            print("[ETL] Удалено строк с пропусками: {}".format(before - len(sales)))

        sales["date"] = pd.to_datetime(sales["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        sales.dropna(subset=["date"], inplace=True)

        n = len(sales)
        valid_pid = set(products["product_id"])
        valid_sid = set(stores["store_id"])
        sales = sales[sales["product_id"].isin(valid_pid) & sales["store_id"].isin(valid_sid)]
        if n - len(sales):
            print("[ETL] Удалено записей с нарушением ссылочной целостности: {}".format(n - len(sales)))

        sales["quantity"] = sales["quantity"].astype(int)
        sales["price"]    = sales["price"].astype(float)
        sales["revenue"]  = sales["quantity"] * sales["price"]

        dt = pd.to_datetime(sales["date"])
        sales["year"]    = dt.dt.year
        sales["month"]   = dt.dt.month
        sales["quarter"] = dt.dt.quarter

        if "customer_id" not in sales.columns:
            np.random.seed(0)
            sales["customer_id"] = np.random.randint(1, 501, size=len(sales))

        print("[ETL] После преобразования: {} записей".format(len(sales)))
        return {"sales": sales, "products": products, "stores": stores}

    def load(self, data):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("DROP TABLE IF EXISTS dim_product")
                conn.execute("""CREATE TABLE dim_product (
                    product_id   INTEGER PRIMARY KEY,
                    product_name TEXT NOT NULL,
                    category     TEXT,
                    supplier     TEXT)""")
                data["products"].to_sql("dim_product", conn,
                                        if_exists="append", index=False)

                conn.execute("DROP TABLE IF EXISTS dim_store")
                conn.execute("""CREATE TABLE dim_store (
                    store_id   INTEGER PRIMARY KEY,
                    city       TEXT NOT NULL,
                    district   TEXT,
                    store_type TEXT)""")
                data["stores"].to_sql("dim_store", conn,
                                      if_exists="append", index=False)

                conn.execute("DROP TABLE IF EXISTS fact_sales")
                conn.execute("""CREATE TABLE fact_sales (
                    sale_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                    date        TEXT NOT NULL,
                    product_id  INTEGER REFERENCES dim_product(product_id),
                    store_id    INTEGER REFERENCES dim_store(store_id),
                    quantity    INTEGER NOT NULL,
                    price       REAL NOT NULL,
                    revenue     REAL NOT NULL,
                    year        INTEGER,
                    month       INTEGER,
                    quarter     INTEGER,
                    customer_id INTEGER)""")
                data["sales"].to_sql("fact_sales", conn,
                                     if_exists="append", index=False)
            print("[ETL] Данные загружены в {}".format(self.db_path))
        except Exception as e:
            print("[ETL][ОШИБКА] Откат транзакции: {}".format(e))
            raise
        finally:
            conn.close()

    def run(self):
        raw  = self.extract()
        data = self.transform(raw)
        self.load(data)


class DataWarehouse:
    REQUIRED_TABLES = {"dim_product", "dim_store", "fact_sales"}

    def __init__(self, db_path):
        self.db_path = db_path

    def check_integrity(self):
        if not os.path.exists(self.db_path):
            print("[DWH] База данных не найдена. Выполните ETL (пункт 1).")
            return False
        conn = sqlite3.connect(self.db_path)
        try:
            existing = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            missing = self.REQUIRED_TABLES - existing
            if missing:
                print("[DWH] Отсутствуют таблицы: {}".format(missing))
                return False
            print("[DWH] Проверка целостности пройдена:")
            for t in self.REQUIRED_TABLES:
                c = conn.execute("SELECT COUNT(*) FROM " + t).fetchone()[0]
                print("       {}: {} записей".format(t, c))
            return True
        finally:
            conn.close()

    def get_checksum(self):
        if not os.path.exists(self.db_path):
            return "N/A"
        with open(self.db_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def query(self, sql, params=()):
        if not os.path.exists(self.db_path):
            raise RuntimeError("База данных не инициализирована. Запустите ETL.")
        conn = sqlite3.connect(self.db_path)
        try:
            return pd.read_sql_query(sql, conn, params=params)
        finally:
            conn.close()


class DataMart:
    def __init__(self, dw):
        self.dw = dw

    def build(self):
        if not os.path.exists(self.dw.db_path):
            raise RuntimeError("База данных не инициализирована. Запустите ETL.")
        conn = sqlite3.connect(self.dw.db_path)
        try:
            with conn:
                conn.execute("DROP VIEW IF EXISTS sales_mart")
                conn.execute("""CREATE VIEW sales_mart AS
                    SELECT
                        f.year, f.quarter, f.month,
                        p.category, p.supplier,
                        p.product_name, p.product_id,
                        s.city, s.store_type, s.store_id,
                        SUM(f.quantity)                     AS total_qty,
                        SUM(f.revenue)                      AS total_revenue,
                        ROUND(SUM(f.revenue)/COUNT(*), 2)   AS avg_check,
                        COUNT(*)                            AS transaction_count,
                        COUNT(DISTINCT f.customer_id)       AS unique_customers
                    FROM fact_sales f
                    JOIN dim_product p ON f.product_id = p.product_id
                    JOIN dim_store   s ON f.store_id   = s.store_id
                    GROUP BY f.year, f.quarter, f.month,
                             p.category, p.supplier, p.product_name, p.product_id,
                             s.city, s.store_type, s.store_id""")
            print("[DataMart] Витрина sales_mart создана.")
        except Exception as e:
            print("[DataMart][ОШИБКА] {}".format(e))
            raise
        finally:
            conn.close()

    def get_mart(self):
        return self.dw.query("SELECT * FROM sales_mart")


class AnalyticsEngine:
    def __init__(self, dw, output_dir):
        self.dw         = dw
        self.output_dir = output_dir

    def abc_analysis(self):
        df = self.dw.query("""
            SELECT p.product_id, p.product_name, SUM(f.revenue) AS total_revenue
            FROM fact_sales f
            JOIN dim_product p ON f.product_id = p.product_id
            GROUP BY p.product_id, p.product_name
            ORDER BY total_revenue DESC""")
        df["revenue_pct"] = df["total_revenue"] / df["total_revenue"].sum() * 100
        groups, cum = [], 0.0
        for pct in df["revenue_pct"]:
            cum += pct
            groups.append("A" if cum <= 80 else ("B" if cum <= 95 else "C"))
        df["abc_group"] = groups
        df["cumulative"] = df["revenue_pct"].cumsum().round(2)
        return df

    def plot_abc(self, df_abc):
        colors     = {"A": "#2ecc71", "B": "#f39c12", "C": "#e74c3c"}
        bar_colors = [colors[g] for g in df_abc["abc_group"]]
        fig, ax    = plt.subplots(figsize=(12, 6))
        ax.bar(df_abc["product_name"], df_abc["total_revenue"],
               color=bar_colors, edgecolor="white", linewidth=0.5)
        ax.set_title("ABC-анализ товаров по выручке", fontsize=14, fontweight="bold")
        ax.set_xlabel("Товар")
        ax.set_ylabel("Выручка (руб.)")
        plt.xticks(rotation=35, ha="right", fontsize=8)
        patches = [mpatches.Patch(color=c, label="Группа "+g) for g, c in colors.items()]
        ax.legend(handles=patches)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:,.0f}".format(x)))
        plt.tight_layout()
        path = os.path.join(self.output_dir, "abc_chart.png")
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def xyz_analysis(self):
        df = self.dw.query("""
            SELECT p.product_id, p.product_name, f.year, f.month,
                   SUM(f.quantity) AS monthly_qty
            FROM fact_sales f
            JOIN dim_product p ON f.product_id = p.product_id
            GROUP BY p.product_id, p.product_name, f.year, f.month""")
        result = []
        for (pid, pname), grp in df.groupby(["product_id","product_name"]):
            mean_qty = grp["monthly_qty"].mean()
            cv = (grp["monthly_qty"].std() / mean_qty * 100) if mean_qty > 0 else 0
            xyz = "X" if cv <= 10 else ("Y" if cv <= 25 else "Z")
            result.append({"product_id": pid, "product_name": pname,
                            "cv_pct": round(cv, 2), "xyz_group": xyz})
        return pd.DataFrame(result).sort_values("cv_pct")

    def abc_xyz_matrix(self, df_abc, df_xyz):
        merged = df_abc[["product_id","product_name","abc_group"]].merge(
            df_xyz[["product_id","xyz_group"]], on="product_id")
        merged["abc_xyz"] = merged["abc_group"] + merged["xyz_group"]
        return merged

    def plot_abc_xyz_heatmap(self, df_matrix):
        pivot = df_matrix.groupby(["abc_group","xyz_group"]).size().unstack(fill_value=0)
        for col in ["X","Y","Z"]:
            if col not in pivot.columns:
                pivot[col] = 0
        pivot = pivot[["X","Y","Z"]].reindex(["A","B","C"])
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGn",
                    linewidths=0.5, ax=ax,
                    cbar_kws={"label": "Кол-во товаров"})
        ax.set_title("Матрица ABC-XYZ", fontsize=13, fontweight="bold")
        ax.set_xlabel("XYZ-группа (стабильность)")
        ax.set_ylabel("ABC-группа (выручка)")
        plt.tight_layout()
        path = os.path.join(self.output_dir, "abc_xyz_heatmap.png")
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def revenue_dynamics(self, category):
        df = self.dw.query("""
            SELECT f.year, f.month, SUM(f.revenue) AS total_revenue
            FROM fact_sales f
            JOIN dim_product p ON f.product_id = p.product_id
            WHERE p.category = ?
            GROUP BY f.year, f.month
            ORDER BY f.year, f.month""", params=(category,))
        if df.empty:
            print("[Analytics] Нет данных для категории: {}".format(category))
            return ""
        df["period"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df["period"], df["total_revenue"], marker="o",
                linewidth=2, color="#3498db", markersize=6)
        ax.fill_between(df["period"], df["total_revenue"], alpha=0.15, color="#3498db")
        ax.set_title("Динамика выручки по месяцам - {}".format(category),
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Период")
        ax.set_ylabel("Выручка (руб.)")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:,.0f}".format(x)))
        plt.tight_layout()
        path = os.path.join(self.output_dir,
                            "revenue_{}.png".format(category.replace(" ","_")))
        plt.savefig(path, dpi=150)
        plt.close()
        return path


class ReportEngine:
    def __init__(self, dw):
        self.dw = dw

    def _last_quarter(self):
        df = self.dw.query("SELECT MAX(year) AS y, MAX(quarter) AS q FROM fact_sales")
        yr, qr = int(df["y"][0]), int(df["q"][0])
        return (yr - 1, 4) if qr == 1 else (yr, qr - 1)

    def report_revenue_by_city(self):
        yr, qr = self._last_quarter()
        return self.dw.query("""
            SELECT s.city, SUM(f.revenue) AS total_revenue, COUNT(*) AS transactions
            FROM fact_sales f JOIN dim_store s ON f.store_id = s.store_id
            WHERE f.year = ? AND f.quarter = ?
            GROUP BY s.city ORDER BY total_revenue DESC""", params=(yr, qr))

    def report_top10_products(self):
        return self.dw.query("""
            SELECT p.product_name, p.category,
                   SUM(f.revenue) AS total_revenue,
                   SUM(f.quantity) AS total_qty
            FROM fact_sales f JOIN dim_product p ON f.product_id = p.product_id
            GROUP BY p.product_id ORDER BY total_revenue DESC LIMIT 10""")

    def report_top5_avg_check(self):
        return self.dw.query("""
            SELECT s.city, s.store_type,
                   ROUND(SUM(f.revenue)/COUNT(*), 2) AS avg_check,
                   COUNT(*) AS transactions
            FROM fact_sales f JOIN dim_store s ON f.store_id = s.store_id
            GROUP BY s.store_id ORDER BY avg_check DESC LIMIT 5""")

    def report_suppliers(self):
        df = self.dw.query("""
            SELECT p.supplier, SUM(f.revenue) AS total_revenue
            FROM fact_sales f JOIN dim_product p ON f.product_id = p.product_id
            GROUP BY p.supplier ORDER BY total_revenue DESC""")
        df["share_pct"] = (df["total_revenue"] / df["total_revenue"].sum() * 100).round(2)
        return df


ABOUT_TEXT = """
+--------------------------------------------------+
|              О ПРОГРАММЕ                         |
+--------------------------------------------------+
| Название  : ИАС «Розничные продажи»              |
| Версия    : 1.0.0                                |
| Дата      : 2026-05-26                           |
| Разработчик: Ефимов Н.С. БИСО-01-21              |
| Реализованные функции:                           |
|  - ETL-процесс (CSV -> SQLite)                   |
|  - Витрина данных sales_mart                     |
|  - ABC-анализ и XYZ-анализ товаров               |
|  - Матрица ABC-XYZ (тепловая карта)              |
|  - Графики динамики выручки                      |
|  - Четыре аналитических отчёта                   |
+--------------------------------------------------+
"""

USER_GUIDE = """
РУКОВОДСТВО ПОЛЬЗОВАТЕЛЯ
========================
1. Запуск: python ias_main.py

2. Главное меню:
   1  - ETL: загрузить данные из CSV в SQLite
   2  - Построить витрину данных (sales_mart)
   3  - ABC-анализ товаров
   4  - XYZ-анализ товаров
   5  - Матрица ABC-XYZ (тепловая карта)
   6  - График динамики выручки по категории
   7  - Отчёт: выручка по городам (последний квартал)
   8  - Отчёт: топ-10 товаров по выручке
   9  - Отчёт: топ-5 магазинов по среднему чеку
   10 - Отчёт: поставщики
   11 - Проверка целостности хранилища
   12 - О программе
   13 - Руководство пользователя
   0  - Выход

3. Требования: Python 3.9+, pandas, matplotlib, seaborn
4. Данные:     каталог data/ (sales.csv, products.csv, stores.csv)
5. Диаграммы:  сохраняются в каталог output/
"""


def print_df(df):
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:,.2f}".format)
    print(df.to_string(index=False))


MENU = """
+----------------------------------------------+
|  1  - ETL: загрузить данные                  |
|  2  - Построить витрину данных               |
|  3  - ABC-анализ                             |
|  4  - XYZ-анализ                             |
|  5  - Матрица ABC-XYZ                        |
|  6  - Динамика выручки по категории          |
|  7  - Отчёт: выручка по городам              |
|  8  - Отчёт: топ-10 товаров                  |
|  9  - Отчёт: топ-5 магазинов                 |
|  10 - Отчёт: поставщики                      |
|  11 - Проверка целостности БД                |
|  12 - О программе                            |
|  13 - Руководство пользователя               |
|  0  - Выход                                  |
+----------------------------------------------+"""


def main():
    print("=" * 55)
    print("  ИАС «Розничные продажи» — Консольное меню")
    print("=" * 55)

    etl = ETLProcessor(DATA_DIR, DB_PATH)
    dw  = DataWarehouse(DB_PATH)
    dm  = DataMart(dw)
    ae  = AnalyticsEngine(dw, OUTPUT_DIR)
    re_ = ReportEngine(dw)

    while True:
        print(MENU)
        choice = input("Выберите пункт: ").strip()

        if choice == "0":
            print("До свидания!")
            break
        elif choice == "1":
            try:
                etl.run()
            except FileNotFoundError as e:
                print("[ОШИБКА] {}".format(e))
        elif choice == "2":
            try:
                dm.build()
            except RuntimeError as e:
                print("[ОШИБКА] {}".format(e))
        elif choice == "3":
            try:
                df = ae.abc_analysis()
                print("\n-- ABC-анализ товаров --")
                print_df(df[["product_name","total_revenue","revenue_pct","cumulative","abc_group"]])
                path = ae.plot_abc(df)
                print("[OK] Диаграмма: {}".format(path))
            except Exception as e:
                print("[ОШИБКА] {}".format(e))
        elif choice == "4":
            try:
                df = ae.xyz_analysis()
                print("\n-- XYZ-анализ товаров --")
                print_df(df[["product_name","cv_pct","xyz_group"]])
            except Exception as e:
                print("[ОШИБКА] {}".format(e))
        elif choice == "5":
            try:
                df_abc = ae.abc_analysis()
                df_xyz = ae.xyz_analysis()
                df_m   = ae.abc_xyz_matrix(df_abc, df_xyz)
                print("\n-- Матрица ABC-XYZ --")
                print_df(df_m[["product_name","abc_group","xyz_group","abc_xyz"]])
                path = ae.plot_abc_xyz_heatmap(df_m)
                print("[OK] Тепловая карта: {}".format(path))
            except Exception as e:
                print("[ОШИБКА] {}".format(e))
        elif choice == "6":
            try:
                cats = dw.query("SELECT DISTINCT category FROM dim_product ORDER BY category")
                print("\nДоступные категории:")
                for i, c in enumerate(cats["category"], 1):
                    print("  {}. {}".format(i, c))
                cat  = input("Введите название категории: ").strip()
                path = ae.revenue_dynamics(cat)
                if path:
                    print("[OK] График: {}".format(path))
            except Exception as e:
                print("[ОШИБКА] {}".format(e))
        elif choice == "7":
            try:
                yr, qr = re_._last_quarter()
                df = re_.report_revenue_by_city()
                print("\n-- Выручка по городам ({} Q{}) --".format(yr, qr))
                print_df(df)
            except Exception as e:
                print("[ОШИБКА] {}".format(e))
        elif choice == "8":
            try:
                print("\n-- Топ-10 товаров по выручке --")
                print_df(re_.report_top10_products())
            except Exception as e:
                print("[ОШИБКА] {}".format(e))
        elif choice == "9":
            try:
                print("\n-- Топ-5 магазинов по среднему чеку --")
                print_df(re_.report_top5_avg_check())
            except Exception as e:
                print("[ОШИБКА] {}".format(e))
        elif choice == "10":
            try:
                print("\n-- Отчёт по поставщикам --")
                print_df(re_.report_suppliers())
            except Exception as e:
                print("[ОШИБКА] {}".format(e))
        elif choice == "11":
            ok = dw.check_integrity()
            if ok:
                print("[OK] Контрольная сумма: {}".format(dw.get_checksum()))
        elif choice == "12":
            print(ABOUT_TEXT)
        elif choice == "13":
            print(USER_GUIDE)
        else:
            print("[!] Неверный выбор.")


if __name__ == "__main__":
    main()
