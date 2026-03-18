import pandas as pd
import json
from tqdm import tqdm
import re
from src.data_processing.duration_preprocessing import add_merge_linkedin_data
from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_datetime64_any_dtype
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Resources:
    pba_types: list  # list of permitted pba types
    country_names_ger: list  # list of country names in German
    hzb_types: list  # list of permitted hzb types
    bl_names: list  # list of Bundesland names
    pba_subjects: list  # list of permitted pba-Subjects
    phd_subjects: list  # list of permitted PhD-subjects
    grades: list  # list of permitted grades
    coops: list  # list of known cooperations
    pub_forms: list  # list of permitted publications forms
    coop_types: list  # list of permitted cooperation types
    promregs: list  # list of possible registration statuses
    new_kostst_list: list[pd.DataFrame]  # list of dataframes containing costnr-data
    kostst_history: pd.DataFrame  # dataframe containing historical costnr-data
    kostst_history_old: (
        pd.DataFrame
    )  # dataframe containing older historical costnr-data
    subject_summary_table: (
        pd.DataFrame
    )  # dataframe containing mapping of subject pdb ->StaBu->Simone's summary
    german_cities: list  # list of German cities
    itl_cities: pd.DataFrame  # list of international cities
    country_alias_dict: (
        dict  # dictionary mapping country name aliases to standard names
    )
    kostst_emeriti_map: dict  # dictionary mapping emeriti costnrs to standard ones
    institute_map: (
        pd.DataFrame
    )  # dataframe mapping institute names to costnrs and lecodes
    birthplace_map: dict  # dictionary of explicit auxiliary mappings between birthplace and counrty not covered by standard options
    country_to_continent: dict  # dictionary mapping countries to their continents
    eu_countries: list  # list of EU countries
    durations: pd.DataFrame  # dataframe containing scraped duration data


@dataclass
class Cols:
    faculties: pd.Series
    genders: pd.Series
    citizenships: pd.Series
    pagin_nr: pd.Series
    first_name: pd.Series
    last_name: pd.Series
    bdate: pd.Series
    reg_stop: pd.Series
    tu_cert: pd.Series
    imma: pd.Series
    hzb_type: pd.Series
    year_first_hzb: pd.Series
    hzb_state: pd.Series
    hzb_bl: pd.Series
    erstimma_staat: pd.Series
    erstimma_uni: pd.Series
    erstimma_uni2: pd.Series
    erstimma_sem: pd.Series
    erstimma_yr: pd.Series
    pba_passed: pd.Series
    pba_state: pd.Series
    pba_uni: pd.Series
    pba_uni2: pd.Series
    pba_type: pd.Series
    pba_subject: pd.Series
    pba_date: pd.Series
    pba_grade: pd.Series
    pba_degree: pd.Series
    phd_prog: pd.Series
    phd_subject: pd.Series
    coop_name: pd.Series
    pub_form: pd.Series
    type_coop: pd.Series
    accept_date: pd.Series
    report_yr_reg: pd.Series
    promreg: pd.Series
    defense_date: pd.Series
    result: pd.Series
    abort: pd.Series
    abort_date: pd.Series
    advisor_firstname: pd.Series
    advisor_lastname: pd.Series
    advisor_costnr: pd.Series
    ref1_type: pd.Series
    ref1_internal: pd.Series
    ref1_firstname: pd.Series
    ref1_lastname: pd.Series
    ref1_costnr: pd.Series
    ref2_type: pd.Series
    ref2_internal: pd.Series
    ref2_firstname: pd.Series
    ref2_lastname: pd.Series
    ref2_costnr: pd.Series
    ref3_internal: pd.Series
    ref3_lastname: pd.Series
    ref3_costnr: pd.Series
    citizenship2: pd.Series
    place_of_birth: pd.Series
    tubesch_nr: pd.Series
    matrnr_student: pd.Series
    matrnr_prom: pd.Series
    type_finance: pd.Series
    finance_start: pd.Series
    finance_end: pd.Series
    process: pd.Series
    program_name: pd.Series
    program_start: pd.Series
    program_end: pd.Series
    prom_degree: pd.Series
    aufl1: pd.Series
    aufl3: pd.Series
    title: pd.Series
    timetable: pd.Series
    abroad_country: pd.Series
    abroad_type: pd.Series
    abroad_from: pd.Series
    abroad_to: pd.Series
    mobility_program: pd.Series
    break_from: pd.Series
    break_to: pd.Series
    chair_firstname: pd.Series
    chair_lastname: pd.Series
    chair_costnr: pd.Series
    ref1_lecode: pd.Series
    # : pd.Seriesadvisor_addinfo
    ref2_secretary: pd.Series
    ref2_lecode: pd.Series
    ref3_type: pd.Series
    ref3_firstname: pd.Series
    ref3_secretary: pd.Series
    ref3_lecode: pd.Series
    open_date: pd.Series
    open_date_application: pd.Series
    review_deadline: pd.Series
    title_de: pd.Series
    title_en: pd.Series
    dis_lng: pd.Series
    grade_latin: pd.Series
    grade_name: pd.Series
    date_publication: pd.Series
    pubnr: pd.Series
    orcid: pd.Series
    abstract: pd.Series
    ub_abstract_date: pd.Series
    certificate_date: pd.Series
    notes: pd.Series
    archive_nr: pd.Series
    linf_planned_date: pd.Series
    linf_exp_date: pd.Series
    connected: pd.Series
    dbaseid: pd.Series
    stala_id: pd.Series
    stala_pseud: pd.Series
    worked_on: pd.Series
    complete: pd.Series
    checked: pd.Series
    email_exists: pd.Series
    yr_defense: pd.Series
    month_defense: pd.Series
    is_german: pd.Series


def load_resources(resource_dir: Path, data_dir: Path) -> Resources:
    """
    Loads all resources from the specified data directory.

    Parameters:
    data_dir: Path to the directory containing the resource files.

    Returns:
    A Resources object containing all the loaded resources.
    """

    def read_csv_as_list(file_path: Path, **kwargs) -> list:
        """
        Reads a CSV file and returns its contents as a list.

        Parameters:
        file_path: Path to the CSV file.
        col_name: if None, reads the header row as list of column names, else reads the specified column as list.
        other kwargs are passed to pd.read_csv.

        Returns:
        A list of strings representing the contents of the CSV file.
        """
        col_name = kwargs.pop("col_name", None)
        if not col_name:
            df = pd.read_csv(file_path, nrows=0)  # reads only header
            return [str(c) for c in df.columns if str(c).strip() != ""]
        else:
            df = pd.read_csv(file_path, **kwargs)
            return df[col_name].astype(str).tolist()

    def read_dict_from_json(file_path: Path) -> dict:
        with open(file_path) as f:
            return json.load(f)

    def read_kostst_files(dir_path: Path) -> list[pd.DataFrame]:
        date_pattern = re.compile(r"^(\d{6}|\d{8})_Kostenstellen\.csv$")
        files_with_dates = []

        for path in dir_path.iterdir():
            if not path.is_file():
                continue

            m = date_pattern.match(path.name)
            if not m:
                continue

            date_str = m.group(1)

            # Normalize YYYYMM → YYYYMM01 so sorting is consistent
            if len(date_str) == 6:
                date_str += "01"

            files_with_dates.append((path, int(date_str)))

        # Sort newest → oldest
        files_with_dates.sort(key=lambda x: x[1], reverse=True)

        # Read CSVs in sorted order
        return [pd.read_csv(path) for path, _ in files_with_dates]

    return Resources(
        pba_types=read_csv_as_list(resource_dir / "pba_types.csv"),
        country_names_ger=read_csv_as_list(resource_dir / "country_names_ger.csv"),
        hzb_types=read_csv_as_list(resource_dir / "hzb_types.csv"),
        bl_names=read_csv_as_list(resource_dir / "bl_names.csv"),
        pba_subjects=read_csv_as_list(resource_dir / "pba_subjects.csv"),
        phd_subjects=read_csv_as_list(resource_dir / "phd_subjects.csv"),
        grades=read_csv_as_list(resource_dir / "grades.csv"),
        coops=read_csv_as_list(resource_dir / "coops.csv"),
        pub_forms=read_csv_as_list(resource_dir / "pub_form.csv"),
        coop_types=read_csv_as_list(resource_dir / "coop_types.csv"),
        promregs=read_csv_as_list(resource_dir / "promregs.csv"),
        new_kostst_list=read_kostst_files(resource_dir / "kostenstellen_tabellen"),
        kostst_history=pd.read_csv(
            resource_dir / "Historie_Kostenstellen_SAP.csv", dtype=str
        ),
        kostst_history_old=pd.read_csv(
            resource_dir / "Historie_Kostenstellen_COB_Stand_31.12.2018.csv", dtype=str
        ),
        subject_summary_table=pd.read_csv(
            resource_dir / "subject_summary.csv", dtype=str, sep=";"
        ),
        german_cities=read_csv_as_list(
            resource_dir / "Liste-Staedte-in-Deutschland.csv",
            dtype=str,
            col_name="Stadt",
        ),
        itl_cities=pd.read_csv(resource_dir / "world_cities_de.csv", dtype=str),
        country_alias_dict=read_dict_from_json(resource_dir / "country_alias.json"),
        kostst_emeriti_map=read_dict_from_json(resource_dir / "emeriti_map.json"),
        institute_map=pd.read_csv(resource_dir / "institute_map.csv", dtype=str),
        birthplace_map=read_dict_from_json(resource_dir / "birthplace_map.json"),
        country_to_continent=read_dict_from_json(
            resource_dir / "country_to_continent.json"
        ),
        eu_countries=read_csv_as_list(
            resource_dir / "eu_countries.csv", dtype=str, col_name="Land"
        ),
        durations=pd.read_csv(data_dir / "durations.csv"),
    )


def get_cols(df):
    """
    Parameters:
    -df: pd.DataFrame containing the raw Promotionsdatenbank Dataset
    Returns:
        Cols: A named Cols dataclass containing the relevant columns from the DataFrame
    """
    return Cols(
        faculties=df.get("Fakultaet", pd.Series(pd.NA, index=df.index, dtype="string")),
        genders=df.get("Geschlecht", pd.Series(pd.NA, index=df.index, dtype="string")),
        citizenships=df.get(
            "Staatsang", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        pagin_nr=df.get("Paginiernr", pd.Series(pd.NA, index=df.index, dtype="string")),
        first_name=df.get("Vorname", pd.Series(pd.NA, index=df.index, dtype="string")),
        last_name=df.get("Nachname", pd.Series(pd.NA, index=df.index, dtype="string")),
        bdate=df.get("Gebdat", pd.Series(pd.NA, index=df.index, dtype="string")),
        reg_stop=df.get(
            "Registrierungsstop", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        tu_cert=df.get("TUBesch", pd.Series(pd.NA, index=df.index, dtype="string")),
        imma=df.get("Imma", pd.Series(pd.NA, index=df.index, dtype="string")),
        hzb_type=df.get("HZB_Art", pd.Series(pd.NA, index=df.index, dtype="string")),
        year_first_hzb=df.get(
            "JahrersteHochschulzugangsberec",
            pd.Series(pd.NA, index=df.index, dtype="string"),
        ),
        hzb_state=df.get("HZB_Staat", pd.Series(pd.NA, index=df.index, dtype="string")),
        hzb_bl=df.get("HZB_BL", pd.Series(pd.NA, index=df.index, dtype="string")),
        erstimma_staat=df.get(
            "Erstimma_Staat",
            pd.Series(pd.NA, index=df.index, dtype="string"),
        ),
        erstimma_uni=df.get(
            "Erstimma_Hochschule",
            pd.Series(pd.NA, index=df.index, dtype="string"),
        ),
        erstimma_uni2=df.get(
            "Erstimma_Hochschule2",
            pd.Series(pd.NA, index=df.index, dtype="string"),
        ),
        erstimma_sem=df.get(
            "Erstimma_Sem",
            pd.Series(pd.NA, index=df.index, dtype="string"),
        ),
        erstimma_yr=df.get(
            "Erstimma_Jahr", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        pba_passed=df.get(
            "pba_Bestanden", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        pba_state=df.get("pba_Staat", pd.Series(pd.NA, index=df.index, dtype="string")),
        pba_uni=df.get(
            "pba_Hochschule", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        pba_uni2=df.get(
            "pba_Hochschule2", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        pba_type=df.get("pba_Art", pd.Series(pd.NA, index=df.index, dtype="string")),
        pba_subject=df.get(
            "pba_Fach", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        pba_date=df.get("pba_Dat", pd.Series(pd.NA, index=df.index, dtype="string")),
        pba_grade=df.get("pba_Note", pd.Series(pd.NA, index=df.index, dtype="string")),
        pba_degree=df.get(
            "pda_akadGrad", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        phd_prog=df.get(
            "PromProgramm", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        phd_subject=df.get(
            "Promfach", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        coop_name=df.get("Koop_Name", pd.Series(pd.NA, index=df.index, dtype="string")),
        pub_form=df.get("Art_Form", pd.Series(pd.NA, index=df.index, dtype="string")),
        type_coop=df.get("Art_Koop", pd.Series(pd.NA, index=df.index, dtype="string")),
        accept_date=df.get(
            "Annahme_Dat", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        report_yr_reg=df.get(
            "Berichtsjahr_reg",
            pd.Series(pd.NA, index=df.index, dtype="string"),
        ),
        promreg=df.get("Promreg", pd.Series(pd.NA, index=df.index, dtype="string")),
        defense_date=df.get(
            "Aussprache_Dat", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        result=df.get("Ergebnis", pd.Series(pd.NA, index=df.index, dtype="string")),
        abort=df.get("Abbruch", pd.Series(pd.NA, index=df.index, dtype="string")),
        abort_date=df.get(
            "Abbruch_Dat", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        advisor_firstname=df.get(
            "Betr_Vorname", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        advisor_lastname=df.get(
            "Betr_Name", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        advisor_costnr=df.get(
            "Betr_Kostst", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        ref1_type=df.get(
            "Gutacht1_Art", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        ref1_internal=df.get(
            "Gutacht1_Intern", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        ref1_firstname=df.get(
            "Gutacht1_VorName", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        ref1_lastname=df.get(
            "Gutacht1_Name", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        ref1_costnr=df.get(
            "Gutacht1_Kostst", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        ref2_type=df.get(
            "Gutacht2_Art", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        ref2_internal=df.get(
            "Gutacht2_Intern", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        ref2_firstname=df.get(
            "Gutacht2_VorName", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        ref2_lastname=df.get(
            "Gutacht2_Name", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        ref2_costnr=df.get(
            "Gutacht2_Kostst", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        ref3_internal=df.get(
            "Gutacht3_Intern", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        ref3_lastname=df.get(
            "Gutacht3_Name", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        ref3_costnr=df.get(
            "Gutacht3_Kostst", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        citizenship2=df.get(
            "Staatsang2", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        place_of_birth=df.get(
            "Gebort", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        tubesch_nr=df.get(
            "TUBesch_Nr", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        matrnr_student=df.get(
            "Matr_Stud", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        matrnr_prom=df.get(
            "Matr_Prom", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        type_finance=df.get(
            "Art_Finanz", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        finance_start=df.get(
            "Finanz_Start", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        finance_end=df.get(
            "Finanz_End", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        process=df.get("Verfahren", pd.Series(pd.NA, index=df.index, dtype="string")),
        program_name=df.get(
            "PromProgr_Name", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        program_start=df.get(
            "PromProgr_Start", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        program_end=df.get(
            "PromProgr_End", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        prom_degree=df.get(
            "PromGrad", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        aufl1=df.get("Auflage1", pd.Series(pd.NA, index=df.index, dtype="string")),
        aufl3=df.get("Auflage3", pd.Series(pd.NA, index=df.index, dtype="string")),
        title=df.get("PromThema", pd.Series(pd.NA, index=df.index, dtype="string")),
        timetable=df.get(
            "Arbeitszeitplan", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        abroad_country=df.get(
            "Ausl_Staat", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        abroad_type=df.get(
            "Ausl_Art", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        abroad_from=df.get(
            "Ausl_Von", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        abroad_to=df.get("Ausl_Bis", pd.Series(pd.NA, index=df.index, dtype="string")),
        mobility_program=df.get(
            "Promotionsmobilitätsprogrammart",
            pd.Series(pd.NA, index=df.index, dtype="string"),
        ),
        break_from=df.get(
            "Break_von", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        break_to=df.get("Beak_bis", pd.Series(pd.NA, index=df.index, dtype="string")),
        chair_firstname=df.get(
            "VorsitzenderVorname", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        chair_lastname=df.get(
            "VorsitzenderNachname", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        chair_costnr=df.get(
            "Vorsitz_Kostst", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        ref1_lecode=df.get(
            "Gutacht1_Sek", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        # advisor_addinfo,
        ref2_secretary=df.get(
            "Gutacht2_Sek", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        ref2_lecode=df.get(
            "Gutacht2_LECode", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        ref3_type=df.get(
            "Gutacht3_Art", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        ref3_firstname=df.get(
            "Gutacht3_VorName", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        ref3_secretary=df.get(
            "Gutacht3_Sek", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        ref3_lecode=df.get(
            "Gutacht3_LECode", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        open_date=df.get("Eröff_Dat", pd.Series(pd.NA, index=df.index, dtype="string")),
        open_date_application=df.get(
            "Eröff_Dat_Antrag", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        review_deadline=df.get(
            "Gutachtenfrist", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        title_de=df.get("Titel_DE", pd.Series(pd.NA, index=df.index, dtype="string")),
        title_en=df.get("Titel_EN", pd.Series(pd.NA, index=df.index, dtype="string")),
        dis_lng=df.get(
            "Diss_Sparche", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        grade_latin=df.get(
            "Note_Latein", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        grade_name=df.get(
            "Note_Name", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        date_publication=df.get(
            "Publikationsfr", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        pubnr=df.get(
            "Publikationsnr", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        orcid=df.get("OrcID", pd.Series(pd.NA, index=df.index, dtype="string")),
        abstract=df.get("Abstrakt", pd.Series(pd.NA, index=df.index, dtype="string")),
        ub_abstract_date=df.get(
            "UbAbstrakt", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        certificate_date=df.get(
            "Urkune_Dat", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        notes=df.get("Bemerkungen", pd.Series(pd.NA, index=df.index, dtype="string")),
        archive_nr=df.get(
            "Archivnummer", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        linf_planned_date=df.get(
            "Linf_planned", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        linf_exp_date=df.get(
            "Linf_exp", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        connected=df.get("Verknüft", pd.Series(pd.NA, index=df.index, dtype="string")),
        dbaseid=df.get("DatenbankID", pd.Series(pd.NA, index=df.index, dtype="string")),
        stala_id=df.get("StalaID", pd.Series(pd.NA, index=df.index, dtype="string")),
        stala_pseud=df.get(
            "StaLaPseudonym", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        worked_on=df.get("bearb", pd.Series(pd.NA, index=df.index, dtype="string")),
        complete=df.get(
            "vollständig", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        checked=df.get(
            "kontrolliert", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        email_exists=df.get(
            "Mail_vorh", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        yr_defense=df.get(
            "Jahr_Aussprache", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        month_defense=df.get(
            "Mon_Aussprache", pd.Series(pd.NA, index=df.index, dtype="string")
        ),
        is_german=df.get("Deutsch", pd.Series(pd.NA, index=df.index, dtype="string")),
    )


def add_n_issue_row(df, field_name, n_issues, n_unique_vals):
    # belongs to checking-stuff, currently broken
    new_row = pd.DataFrame(
        [{"col_name": field_name, "n_issues": n_issues, "n_unique_vals": n_unique_vals}]
    )
    df = pd.concat([df, new_row], ignore_index=True)
    return df


def issue_table_to_dict(df, dictionary, field_name, matches, field_name_native):
    # belongs to checking-stuff, currently broken
    if not field_name_native:
        field_name_native = [field_name]
    elif isinstance(field_name_native, str):
        field_name_native = [field_name_native]
    saved_fields = ["Vorname", "Nachname"]
    for fld_name in field_name_native:
        saved_fields.append(fld_name)
    problem_rows = df[matches]
    dictionary[field_name] = problem_rows[saved_fields]


def commit_issue_data(
    df,
    issues_table,
    issue_dict,
    vals_dict,
    field_name,
    n_issues,
    matches,
    mask_table,
    field_name_native=[],
    unique_by_field=None,
):
    # belongs to checking-stuff, currently broken
    if field_name_native:
        if type(field_name_native) is list and len(field_name_native) > 1:
            assert unique_by_field is not None, (
                "If field_name_native is a list of more than one element, unique_by_field must be provided."
            )
        elif type(field_name_native) is list:
            unique_by_field = field_name_native[0]
        else:
            unique_by_field = field_name_native
    else:
        unique_by_field = field_name
    unique_problem_values = df[matches][unique_by_field].unique()
    n_unique_vals = len(unique_problem_values)
    vals_dict[field_name] = [n_unique_vals, unique_problem_values]

    issues_table = add_n_issue_row(issues_table, field_name, n_issues, n_unique_vals)

    issue_table_to_dict(
        df, issue_dict, field_name, matches, field_name_native=field_name_native
    )
    mask_table[field_name] = matches
    return issues_table


def build_cleaned_table(
    df, keep_original=False, resource_path=Path("resources"), data_path=Path("data")
):
    """
    Build a cleaned version of the input DataFrame.
    Parameters
    -df : pd.DataFrame
        The input DataFrame to clean.
    -keep_original : bool, optional
        Whether to keep the original columns in the cleaned DataFrame.

    Returns
    pd.DataFrame
        The cleaned DataFrame.
        TODO: description of all columns
    """
    resources = load_resources(resource_path, data_path)
    cols = get_cols(df)

    # each key is the name of the cleaned column. Its value
    # is a tuple containing the cleaning function, its args and its kwargs
    cleaning_dict = {
        "pagination_nr": (copy_col, (cols.pagin_nr,), {}),
        "faculty": (copy_col, (cols.faculties,), {}),
        "first_name": (copy_col, (cols.first_name,), {}),
        "last_name": (copy_col, (cols.last_name,), {}),
        "full_name": (get_fullname, (cols.first_name, cols.last_name), {}),
        "gender": (copy_col, (cols.genders,), {"na": "d"}),
        "birth_date": (copy_col, (cols.bdate,), {}),
        "birth_year": (extract_year, (cols.bdate,), {}),
        "birth_month": (extract_month, (cols.bdate,), {}),
        "birth_day": (extract_day, (cols.bdate,), {}),
        "citizenship": (
            alias_country_names,
            (cols.citizenships, resources.country_alias_dict),
            {},
        ),
        "citizenship2": (
            alias_country_names,
            (cols.citizenship2, resources.country_alias_dict),
            {},
        ),
        "has_german_cship": (
            check_german_cship,
            (cols.citizenships, cols.citizenship2),
            {},
        ),
        "has_eu_cship": (
            check_eu_cship,
            (cols.citizenships, cols.citizenship2, resources.eu_countries),
            {},
        ),
        "survey_stopped": (yn_to_bool, (cols.reg_stop,), {}),
        "tu_employed": (yn_to_bool, (cols.tu_cert,), {}),
        "immatriculated": (yn_to_bool, (cols.imma,), {}),
        "preacademic_degree": (copy_col, (cols.hzb_type,), {}),
        "year_first_preac_degree": (clean_year, (cols.year_first_hzb,), {}),
        "preac_state": (
            alias_country_names,
            (cols.hzb_state, resources.country_alias_dict),
            {},
        ),
        "preac_state_in_eu": (check_eu, (cols.hzb_state, resources.eu_countries), {}),
        "preac_bl": (copy_col, (cols.hzb_bl,), {}),
        "firstimma_state": (
            alias_country_names,
            (cols.erstimma_staat, resources.country_alias_dict),
            {},
        ),
        "firstimma_eu": (check_eu, (cols.erstimma_staat, resources.eu_countries), {}),
        "firstimma_winter_start": (sem_to_bool, (cols.erstimma_sem,), {}),
        "firstimma_year": (clean_year, (cols.erstimma_yr,), {}),
        "pba_passed": (yn_to_bool, (cols.pba_passed,), {}),
        "pba_state": (
            alias_country_names,
            (cols.pba_state, resources.country_alias_dict),
            {},
        ),
        "pba_type": (copy_col, (cols.pba_type,), {}),
        "pba_subject": (copy_col, (cols.pba_subject,), {}),
        "pba_date": (copy_col, (cols.pba_date,), {}),
        "pba_year": (extract_year, (cols.pba_date,), {}),
        "pba_grade": (grade_to_float, (cols.pba_grade,), {}),
        "pba_degree": (copy_col, (cols.pba_degree,), {}),
        "phd_program": (yn_to_bool, (cols.phd_prog,), {}),
        "phd_subject": (copy_col, (cols.phd_subject,), {}),
        "subject_group": (
            subject_to_grouping,
            (cols.phd_subject, resources.subject_summary_table),
            {"transform_column": "summary_simone"},
        ),
        "subject_group_stala": (
            subject_to_grouping,
            (
                cols.phd_subject,
                resources.subject_summary_table,
            ),
            {"transform_column": "stala"},
        ),
        "subject_group_category_stala": (
            subject_to_grouping,
            (
                cols.phd_subject,
                resources.subject_summary_table,
            ),
            {"transform_column": "stala_groups"},
        ),
        "subject_groupcat_stala_ind": (
            subject_to_grouping,
            (
                cols.phd_subject,
                resources.subject_summary_table,
            ),
            {"transform_column": "stala_groups_index"},
        ),
        "coop_name": (copy_col, (cols.coop_name,), {}),
        "publication_form": (copy_col, (cols.pub_form,), {}),
        "coop_type": (copy_col, (cols.type_coop,), {}),
        "acceptance_date": (copy_col, (cols.accept_date,), {}),
        "acceptance_year": (extract_year, (cols.accept_date,), {}),
        "report_year": (copy_col, (cols.report_yr_reg,), {}),
        "report_status": (copy_col, (cols.promreg,), {}),
        "defense_date": (get_defense_date, (cols.defense_date,), {}),
        "defense_year": (get_defense_year, (cols.defense_date,), {}),
        "successful_defense": (defense_to_bool, (cols.result,), {}),
        "aborted": (yn_to_bool, (cols.abort,), {}),
        "abort_date": (copy_col, (cols.abort_date,), {}),
        "abort_year": (extract_year, (cols.abort_date,), {}),
        "advisor_firstname": (copy_col, (cols.advisor_firstname,), {}),
        "advisor_lastname": (copy_col, (cols.advisor_lastname,), {}),
        "advisor_fullname": (
            get_fullname,
            (cols.advisor_firstname, cols.advisor_lastname),
            {},
        ),
        "advisor_costnr": (
            clean_costnr,
            (
                cols.advisor_costnr,
                cols.advisor_firstname,
                cols.advisor_lastname,
                pd.Series(["Ja"] * len(cols.advisor_costnr)),
                resources.kostst_emeriti_map,
                resources.new_kostst_list,
                resources.kostst_history,
                resources.kostst_history_old,
            ),
            {"use_successor": True, "verbose": False},
        ),
        "ref1_type": (copy_col, (cols.ref1_type,), {}),
        "ref1_internal": (yn_to_bool, (cols.ref1_internal,), {}),
        "ref1_firstname": (copy_col, (cols.ref1_firstname,), {}),
        "ref1_lastname": (copy_col, (cols.ref1_lastname,), {}),
        "ref1_fullname": (get_fullname, (cols.ref1_firstname, cols.ref1_lastname), {}),
        "ref1_costnr": (
            clean_costnr,
            (
                cols.ref1_costnr,
                cols.ref1_firstname,
                cols.ref1_lastname,
                cols.ref1_internal,
                resources.kostst_emeriti_map,
                resources.new_kostst_list,
                resources.kostst_history,
                resources.kostst_history_old,
            ),
            {"use_successor": True, "verbose": False},
        ),  # same as advisor_costnr
        "ref2_type": (copy_col, (cols.ref2_type,), {}),
        "ref2_internal": (yn_to_bool, (cols.ref2_internal,), {}),
        "ref2_firstname": (copy_col, (cols.ref2_firstname,), {}),
        "ref2_lastname": (copy_col, (cols.ref2_lastname,), {}),
        "ref2_fullname": (get_fullname, (cols.ref2_firstname, cols.ref2_lastname), {}),
        "ref2_costnr": (
            clean_costnr,
            (
                cols.ref2_costnr,
                cols.ref2_firstname,
                cols.ref2_lastname,
                cols.ref2_internal,
                resources.kostst_emeriti_map,
                resources.new_kostst_list,
                resources.kostst_history,
                resources.kostst_history_old,
            ),
            {},
        ),  # same as advisor_costnr
        "ref3_type": (copy_col, (cols.ref3_type,), {}),
        "ref3_internal": (yn_to_bool, (cols.ref3_internal,), {}),
        "ref3_lastname": (copy_col, (cols.ref3_lastname,), {}),
        "ref3_firstname": (copy_col, (cols.ref3_firstname,), {}),
        "ref3_fullname": (get_fullname, (cols.ref3_firstname, cols.ref3_lastname), {}),
        "ref3_costnr": (
            clean_costnr,
            (
                cols.ref3_costnr,
                cols.ref3_firstname,
                cols.ref3_lastname,
                cols.ref3_internal,
                resources.kostst_emeriti_map,
                resources.new_kostst_list,
                resources.kostst_history,
                resources.kostst_history_old,
            ),
            {},
        ),  # same as advisor_costnr
        "birthplace": (copy_col, (cols.place_of_birth,), {}),  # implement flowchart
        "birth_country": (
            get_birth_country,
            (
                cols.place_of_birth,
                resources.birthplace_map,
                resources.country_alias_dict,
                resources.country_names_ger,
                resources.german_cities,
                resources.itl_cities,
            ),
            {},
        ),
        "type_financing": (copy_col, (cols.type_finance,), {}),
        "financing_start": (copy_col, (cols.finance_start,), {}),
        "financing_end": (copy_col, (cols.finance_end,), {}),
        "process": (yn_to_bool, (cols.process,), {}),
        "program_name": (copy_col, (cols.program_name,), {}),
        "program_start": (copy_col, (cols.program_start,), {}),
        "program_end": (copy_col, (cols.program_end,), {}),
        "in_program": (
            check_nonempty,
            (cols.program_name, cols.program_start, cols.program_end),
            {},
        ),
        "prom_degree": (copy_col, (cols.prom_degree,), {}),
        "stip_1": (yn_to_bool, (cols.aufl1,), {}),
        "stip_3": (yn_to_bool, (cols.aufl3,), {}),
        "topic": (copy_col, (cols.title,), {}),
        "chair_firstname": (copy_col, (cols.chair_firstname,), {}),
        "chair_lastname": (copy_col, (cols.chair_lastname,), {}),
        "chair_fullname": (
            get_fullname,
            (cols.chair_firstname, cols.chair_lastname),
            {},
        ),
        "chair_costnr": (
            clean_costnr,
            (
                cols.chair_costnr,
                cols.chair_firstname,
                cols.chair_lastname,
                pd.Series(["Ja"] * len(cols.chair_costnr)),
                resources.kostst_emeriti_map,
                resources.new_kostst_list,
                resources.kostst_history,
                resources.kostst_history_old,
            ),
            {},
        ),  # same as advisor_costnr
        "adv_costnr_original": (copy_col, (cols.advisor_costnr,), {}),
        "ref1_costnr_original": (copy_col, (cols.ref1_costnr,), {}),
        "ref2_costnr_original": (copy_col, (cols.ref2_costnr,), {}),
        "ref3_costnr_original": (copy_col, (cols.ref3_costnr,), {}),
        "chair_costnr_original": (copy_col, (cols.chair_costnr,), {}),
        "open_application_date": (copy_col, (cols.open_date_application,), {}),
        "open_date": (copy_col, (cols.open_date,), {}),
        "review_deadline": (copy_col, (cols.review_deadline,), {}),
        "title_de": (copy_col, (cols.title_de,), {}),
        "title_en": (copy_col, (cols.title_en,), {}),
        "dissertation_language": (copy_col, (cols.dis_lng,), {}),
        "grade_latin": (copy_col, (cols.grade_latin,), {}),
        "grade_name": (copy_col, (cols.grade_name,), {}),
        "grade_num": (grade_to_float, (cols.grade_name,), {}),
        "dbase_id": (copy_col, (cols.dbaseid,), {}),
        "stala_id": (copy_col, (cols.stala_id,), {}),
        "stala_pseud": (copy_col, (cols.stala_pseud,), {}),
        "worked_on": (copy_col, (cols.worked_on,), {}),
        "complete": (yn_to_bool, (cols.complete,), {}),
        "checked": (yn_to_bool, (cols.checked,), {}),
        "mail_exists": (num_to_bool, (cols.email_exists,), {}),
        "year_defense": (clean_year, (cols.yr_defense,), {}),
        "month_defense": (copy_col, (cols.month_defense,), {}),
        "lang_is_german": (num_to_bool, (cols.is_german,), {}),
        "matrnr_student": (copy_col, (cols.matrnr_student,), {}),
        "matrnr_prom": (copy_col, (cols.matrnr_prom,), {}),
    }

    # same format as the first dictionary, but now the cleaning functions can also be applied to
    # columns cleaned in the previous step. When an arg is a tuple ("col", col_name), the column
    # named "col_name" created in the previous step is used.
    secondary_cleaning_dict = {
        "noacc_noabrt_nodef_check": (
            noacc_noabrt_nodef_check,
            (("col", "acceptance_date"), ("col", "aborted"), ("col", "defense_date")),
            {},
        ),
        "cship_category": (
            cship_category,
            (("col", "has_german_cship"), ("col", "has_eu_cship")),
            {},
        ),
        "advisor_institute_nr": (get_institute_nr, (("col", "advisor_costnr"),), {}),
        "advisor_institute_name": (
            get_institute_name,
            (("col", "advisor_costnr"), resources.institute_map),
            {},
        ),
        "advisor_lecode": (
            get_lecode,
            (("col", "advisor_costnr"), resources.institute_map),
            {},
        ),
        "ref1_institute_nr": (get_institute_nr, (("col", "ref1_costnr"),), {}),
        "ref1_institute_name": (
            get_institute_name,
            (("col", "ref1_costnr"), resources.institute_map),
            {},
        ),
        "ref1_lecode": (
            get_lecode,
            (("col", "ref1_costnr"), resources.institute_map),
            {},
        ),
        "ref2_institute_nr": (get_institute_nr, (("col", "ref2_costnr"),), {}),
        "ref2_institute_name": (
            get_institute_name,
            (("col", "ref2_costnr"), resources.institute_map),
            {},
        ),
        "ref2_lecode": (
            get_lecode,
            (("col", "ref2_costnr"), resources.institute_map),
            {},
        ),
        "ref3_institute_nr": (get_institute_nr, (("col", "ref3_costnr"),), {}),
        "ref3_institute_name": (
            get_institute_name,
            (("col", "ref3_costnr"), resources.institute_map),
            {},
        ),
        "ref3_lecode": (
            get_lecode,
            (("col", "ref3_costnr"), resources.institute_map),
            {},
        ),
        "chair_institute_nr": (get_institute_nr, (("col", "chair_costnr"),), {}),
        "chair_institute_name": (
            get_institute_name,
            (("col", "chair_costnr"), resources.institute_map),
            {},
        ),
        "chair_lecode": (
            get_lecode,
            (("col", "chair_costnr"), resources.institute_map),
            {},
        ),
        "firstimma_uni": (
            extract_uni,
            (cols.erstimma_uni, cols.erstimma_uni2, ("col", "firstimma_state")),
            {},
        ),
        "firstimma_uni_is_german": (
            uni_is_german,
            (cols.erstimma_uni, cols.erstimma_uni2, ("col", "firstimma_state")),
            {},
        ),
        "pba_uni": (
            extract_uni,
            (cols.pba_uni, cols.pba_uni2, ("col", "pba_state")),
            {},
        ),
        "pba_uni_is_german": (
            uni_is_german,
            (cols.pba_uni, cols.pba_uni2, ("col", "pba_state")),
            {},
        ),
        "pba_uni_is_tub": (uni_is_tub, (cols.pba_uni,), {}),
        "citizenship_continent": (
            map_country_to_continent,
            (("col", "citizenship"), resources.country_to_continent),
            {},
        ),
        "pba_uni_is_fh": (
            uni_is_fh,
            (cols.pba_uni,),
            {},
        ),
        "pba_uni_in_eu": (check_eu, (cols.pba_state, resources.eu_countries), {}),
        "citizenship2_continent": (
            map_country_to_continent,
            (("col", "citizenship2"), resources.country_to_continent),
            {},
        ),
        "preac_continent": (
            map_country_to_continent,
            (("col", "preac_state"), resources.country_to_continent),
            {},
        ),
        "firstimma_continent": (
            map_country_to_continent,
            (("col", "firstimma_state"), resources.country_to_continent),
            {},
        ),
        "birth_continent": (
            map_country_to_continent,
            (("col", "birth_country"), resources.country_to_continent),
            {},
        ),
        "born_in_eu": (
            check_eu,
            (("col", "birth_country"), resources.eu_countries),
            {},
        ),
        "pba_grade_justnum": (grade_to_justnum, (("col", "pba_grade"),), {}),
        "grade_justnum": (grade_to_justnum, (("col", "grade_num"),), {}),
    }

    # same as secondary, but necessary to operate on columns created in previous step
    ternary_cleaning_dict = {
        "institute_name": (
            fetch_first_existing,
            (
                ("col", "advisor_institute_name"),
                ("col", "ref1_institute_name"),
                ("col", "ref2_institute_name"),
                ("col", "ref3_institute_name"),
                ("col", "chair_institute_name"),
            ),
            {},
        ),
        "lecode": (
            fetch_first_existing,
            (
                ("col", "advisor_lecode"),
                ("col", "ref1_lecode"),
                ("col", "ref2_lecode"),
                ("col", "ref3_lecode"),
                ("col", "chair_lecode"),
            ),
            {},
        ),
        "d_acceptance": (
            create_d_col,
            (("col", "acceptance_date"), ("col", "defense_date")),
            {},
        ),
        "d_pba": (create_d_col, (("col", "pba_date"), ("col", "defense_date")), {}),
        "pba_category": (
            categorize_uniloc,
            (
                ("col", "pba_uni_is_tub"),
                ("col", "pba_uni_is_fh"),
                ("col", "pba_uni_is_german"),
                ("col", "pba_uni_in_eu"),
            ),
            {},
        ),
    }

    # we specify a list for merging in raw columns from the original table. Each element of the list
    # is a tuple containing a name (for the new table), old col as a Series and the reference col in
    # the new table to compare it with. Columns will be suffixed '_orig', and boolean columns
    # suffixed 'orig_check' will be True when there is a mismatch in the respective row with the reference column.
    original_merge_list = [
        ("Staatsang", cols.citizenships, "citizenship"),
        ("Staatsang2", cols.citizenship2, "citizenship2"),
        (
            "JahrersteHochschulzugangsberechtigung",
            cols.year_first_hzb,
            "year_first_preac_degree",
        ),
        ("HZB_Staat", cols.hzb_state, "preac_state"),
        ("Erstimma_Staat", cols.erstimma_staat, "firstimma_state"),
        ("Erstimma_Jahr", cols.erstimma_yr, "firstimma_year"),
        ("pba_Staat", cols.pba_state, "pba_state"),
        ("defense_date", cols.defense_date, "defense_date"),
    ]

    full_dset_list = [
        (remove_duplicates, ()),  # should be first in the list
        (
            merge_linf_duration,
            (data_path / "20250821_PromAbschluesse2000_2020_TUB.csv",),
        ),
        (add_merge_linkedin_data, (data_path / "durations.csv",)),
        (remove_duplicates, ()),  # added as last step due to potential reduplication
    ]

    primary_cols = {}
    for key in tqdm(cleaning_dict, desc="Cleaning columns"):
        tqdm.write(f"Processing column: {key}")
        func, args, kwargs = cleaning_dict[key]
        primary_cols[key] = func(*args, **kwargs)

    # Build cleaned_df from primary columns
    cleaned_df = pd.DataFrame(primary_cols)

    # Now compute secondary columns
    secondary_cols = {}
    for key in tqdm(secondary_cleaning_dict, desc="Processing secondary computations"):
        tqdm.write(f"Processing column: {key}")
        func, args, kwargs = secondary_cleaning_dict[key]
        resolved_args = resolve_args(cleaned_df, args)
        secondary_cols[key] = func(*resolved_args, **kwargs)

    # Add secondary columns all at once
    cleaned_df = pd.concat([cleaned_df, pd.DataFrame(secondary_cols)], axis=1)

    # Ternary columns
    ternary_cols = {}
    for key in tqdm(ternary_cleaning_dict, desc="Processing ternary computations"):
        tqdm.write(f"Processing column: {key}")
        func, args, kwargs = ternary_cleaning_dict[key]
        resolved_args = resolve_args(cleaned_df, args)
        ternary_cols[key] = func(*resolved_args, **kwargs)

    cleaned_df = pd.concat([cleaned_df, pd.DataFrame(ternary_cols)], axis=1)

    if keep_original:
        original_merge_cols = {}
        for item in original_merge_list:
            tqdm.write(f"Inlcuding original column: {item[0]}")
            col, col_check = copy_and_compare(item[1], cleaned_df[item[2]])
            original_merge_cols[item[0] + "_orig"] = col
            original_merge_cols[item[0] + "_orig_check"] = col_check
        cleaned_df = pd.concat([cleaned_df, pd.DataFrame(original_merge_cols)], axis=1)

    for function, args in full_dset_list:
        tqdm.write(f"Processing full dataset function: {function.__name__}")
        cleaned_df = function(cleaned_df, *args)

    for col in cleaned_df.columns:
        dtype_col = cleaned_df[col].dtype
        match dtype_col.name:
            case "object":
                cleaned_df[col] = cleaned_df[col].astype("string")
            case "bool":
                cleaned_df[col] = cleaned_df[col].astype("boolean")
            case "int64":
                cleaned_df[col] = cleaned_df[col].astype("Int64")
            case "float64":
                cleaned_df[col] = cleaned_df[col].astype("Float64")

    dtype_dict = cleaned_df.dtypes.apply(lambda x: x.name).to_dict()

    return cleaned_df, dtype_dict


# def subject_to_grouping(subject_col, subject_summary_table, stala=False):
#     # build a mapping from Fach laut PDB -> Zusammenfassung
#     if not stala:
#         mapping = subject_summary_table.set_index("Fach laut PDB")["Zusammenfassung 1"]
#     else:
#         mapping = subject_summary_table.set_index("Fach laut PDB")["Fach StaBu"]
#
#     s = subject_col.astype("string")
#     is_blank = s.isna() | (s == "")
#
#     mapped = s.map(mapping)  # vectorized lookup
#     mapped[is_blank] = pd.NA  # preserve blanks as <NA>
#     mapped[~is_blank & mapped.isna()] = "Sonstige"  # not blank but no match
#
#     return mapped.astype("string")
#


def subject_to_grouping(
    subject_col, subject_summary_table, transform_column="summary_simone"
):
    match transform_column:
        case "summary_simone":
            colname = "Zusammenfassung 1"
        case "stala":
            colname = "Fach StaBu"
        case "stala_groups":
            colname = "Fächergruppe"
        case "stala_groups_index":
            colname = "group_index"
        case _:
            raise ValueError(f"Invalid transform_column value: {transform_column}")

    mapping = subject_summary_table.set_index("Fach laut PDB")[colname]

    s = subject_col.astype("string")
    is_blank = s.isna() | (s == "")

    mapped = s.map(mapping)  # vectorized lookup
    mapped[is_blank] = pd.NA  # preserve blanks as <NA>
    mapped[~is_blank & mapped.isna()] = "Sonstige"  # not blank but no match

    return mapped.astype("string")


def categorize_uniloc(is_tub, is_fh, is_german, in_eu):
    ret_series = pd.Series(index=is_tub.index, dtype="string")
    ret_series[in_eu] = "EU"
    ret_series[is_german] = "GER"
    ret_series[~is_german & ~in_eu] = "NEU"
    ret_series[is_tub] = "TUB"
    ret_series[is_fh] = "FH"
    return ret_series


def mod_defense_date(defense_date):
    mask = defense_date == "18-10-0218"
    defense_date[mask] = "18-10-2018"
    mask = defense_date == "12-10-0018"
    defense_date[mask] = "12-10-2018"
    mask = defense_date == "05-02-0018"
    defense_date[mask] = "05-02-2018"
    return defense_date


def cship_category(has_german_cship, has_eu_cship):
    cship_cat = pd.Series(index=has_german_cship.index, dtype="string")
    cship_cat[has_german_cship] = "German"
    cship_cat[has_eu_cship & ~has_german_cship] = "Non-German EU"
    cship_cat[~has_german_cship & ~has_eu_cship] = "Non-EU"
    return cship_cat


def create_d_col(convert_col, ref_col):
    convert_col = pd.to_datetime(convert_col, format="%d-%m-%Y")
    ref_col = pd.to_datetime(ref_col, format="%d-%m-%Y")

    diff = ref_col - convert_col
    diff = diff.dt.days / 365

    return diff


def get_defense_date(defense_date):
    return copy_col(mod_defense_date(defense_date))


def get_defense_year(defense_date):
    return extract_year(mod_defense_date(defense_date))


def remove_duplicates(df):
    # for now only pagination_nr
    dupmask = df.duplicated(subset=["pagination_nr"])
    dup_pagins = df[dupmask]["pagination_nr"].unique()
    print(f"Found duplicate pagination numbers: {dup_pagins}")
    return df[~dupmask].reset_index(drop=True)


def merge_linf_duration(df, linf_csv_path):
    linf_data = pd.read_csv(
        linf_csv_path,
        sep=";",
        dtype={"Gebmon": str, "Gebjahr": str},
    )

    linf_data_tmp = linf_data[
        [
            "NachnamePromovend",
            "VornamePromovend",
            "GebDatTag",
            "Gebmon",
            "Gebjahr",
            "PromDauer",
        ]
    ].copy()

    linf_data_tmp.loc[:, "birth_day"] = pd.to_numeric(
        linf_data_tmp["GebDatTag"], errors="coerce"
    )
    linf_data_tmp.loc[:, "birth_month"] = pd.to_numeric(
        linf_data_tmp["Gebmon"], errors="coerce"
    )
    linf_data_tmp.loc[:, "birth_year"] = pd.to_numeric(
        linf_data_tmp["Gebjahr"], errors="coerce"
    )

    linf_data_tmp.loc[linf_data_tmp["birth_day"] > 31, "birth_day"] = pd.NA
    linf_data_tmp.loc[linf_data_tmp["birth_day"] < 1, "birth_day"] = pd.NA
    linf_data_tmp.loc[linf_data_tmp["birth_month"] > 12, "birth_month"] = pd.NA
    linf_data_tmp.loc[linf_data_tmp["birth_month"] < 1, "birth_month"] = pd.NA
    linf_data_tmp.loc[linf_data_tmp["birth_month"] < 1, "birth_month"] = pd.NA
    linf_data_tmp.loc[linf_data_tmp["birth_year"] < 1900, "birth_year"] = pd.NA
    linf_data_tmp.loc[linf_data_tmp["birth_year"] > 2222, "birth_year"] = pd.NA

    linf_data_tmp.loc[linf_data_tmp["PromDauer"] < 0, "PromDauer"] = pd.NA
    linf_data_tmp.loc[linf_data_tmp["PromDauer"] == 99, "PromDauer"] = pd.NA

    # linf_data_tmp["first_name"] = linf_data_tmp["VornamePromovend"].copy()
    # linf_data_tmp["last_name"] = linf_data_tmp["NachnamePromovend"].copy()
    linf_data_tmp.loc[:, "first_name"] = linf_data_tmp["VornamePromovend"].copy()
    linf_data_tmp.loc[:, "last_name"] = linf_data_tmp["NachnamePromovend"].copy()

    linf_data_tmp.drop(
        ["GebDatTag", "Gebmon", "Gebjahr", "VornamePromovend", "NachnamePromovend"],
        axis=1,
        inplace=True,
    )
    linf_data_tmp = linf_data_tmp[~linf_data_tmp["PromDauer"].isna()]

    mg = pd.merge(
        linf_data_tmp,
        df,
        on=["first_name", "last_name", "birth_year", "birth_month", "birth_day"],
        how="right",
    )
    mg.rename(columns={"PromDauer": "duration_linf"}, inplace=True)
    return mg


def fetch_first_existing(*args):
    ret = pd.Series(index=args[0].index, dtype="string")
    to_change_mask = pd.Series([True] * len(ret), index=ret.index)
    for col in args:
        values_exist_mask = col.notna() & (col != "")
        change_mask = to_change_mask & values_exist_mask

        ret[change_mask] = col[change_mask]
        to_change_mask[change_mask] = False

    return ret


def uni_is_fh(uni):
    pattern = re.compile(r"(?i)(?<![A-ZÄÖÜa-zäöü])FH(?![A-ZÄÖÜa-zäöü])|Fachhochschule")
    return uni.str.contains(pattern, na=False)


def uni_is_tub(uni):
    return uni == "Berlin, TU"


def grade_to_justnum(grades):
    # make grades pd.NA where its a string and keep the other values. Values are either float or string or pd.NA
    ret = grades.copy()
    mask = ret.apply(lambda x: isinstance(x, str) and not x.isnumeric())
    ret[mask] = pd.NA
    return ret.astype("Float64")


def check_eu(countries, eu_countries):
    return countries.isin(eu_countries)


def check_eu_cship(countries, countries2, eu_countries):
    return countries.isin(eu_countries) | countries2.isin(eu_countries)


def extract_year(date):
    mask_nonempty = date.notna() & (date != "")
    ret = date.copy()
    ret[mask_nonempty] = date[mask_nonempty].str[-4:]
    ret[~mask_nonempty] = pd.NA
    return ret.astype("Int64")


def extract_month(date):
    mask_nonempty = date.notna() & (date != "")
    ret = date.copy()
    ret[mask_nonempty] = date[mask_nonempty].str[3:5].str.lstrip("0")
    ret[~mask_nonempty] = pd.NA
    return ret.astype("Int64")


def extract_day(date):
    mask_nonempty = date.notna() & (date != "")
    ret = date.copy()
    ret[mask_nonempty] = date[mask_nonempty].str[:2].str.lstrip("0")
    ret[~mask_nonempty] = pd.NA
    return ret.astype("Int64")


def resolve_args(table, args):
    """
    Parameters:
    - table: The DataFrame containing the data.
    - args: A list of arguments to resolve.

    Returns the correct arguments for other functions in a generator. For each arg in args, if it is a tuple with the second element being 'col',
    the first argument is treated as a column name and the corresponding column from the table is returned. Otherwise, the argument is returned as is.
    """
    return (
        table[arg[1]] if isinstance(arg, tuple) and arg[0] == "col" else arg
        for arg in args
    )


def get_institute_nr(costnr_col):
    return costnr_col.str[:4]


def get_institute_name(costnr, institute_map):
    institute_nr = get_institute_nr(costnr)
    return institute_nr.map(institute_map.set_index("Kostenstelle")["Bezeichnung"])


def load_data(data_csv, dtypes_json):
    with open(dtypes_json) as f:
        dtypes = json.load(f)
    return pd.read_csv(
        data_csv,
        dtype=dtypes,
        keep_default_na=False,
        na_values=[""],
        dtype_backend="pyarrow",
    )


def get_lecode(costnr, institute_map):
    insitute_nr = get_institute_nr(costnr)
    return insitute_nr.map(institute_map.set_index("Kostenstelle")["Lehreinheit"])


def get_birth_country(
    birthplace: pd.Series,
    birthplace_map: dict,
    country_alias_dict: dict,
    country_names_ger: pd.Series,
    german_cities: pd.Series,
    itl_cities: pd.Series,
) -> pd.Series:
    """
    Return inferred birth countries based on birthplace, using German and international mappings. This function is unoptimized and quite slow, but because
    it is called rarely and the dataset is not huge, this is acceptable for now. Might improve at a later point.
    Parameters:
    - birthplace: Series of birthplace strings.
    - birthplace_map: Dictionary mapping birthplace strings to country names.
    - country_alias_dict: Dictionary mapping country names to their aliases.
    - country_names_ger: Series of German country names.
    - german_cities: Series of German city names.
    - itl_cities: Series of international city names.
    Returns:
    Series of inferred birth countries.

    """
    ret = pd.Series(index=birthplace.index, dtype="string")

    for i, bplace in birthplace.items():
        if pd.isna(bplace):
            ret.loc[i] = pd.NA
            continue
        elif bplace in birthplace_map:
            ret.loc[i] = birthplace_map[bplace]
            continue

        country = map_to_country_ger(bplace, country_names_ger)
        if pd.isna(country):
            country = map_city_to_country_ger(bplace, german_cities)
        if pd.isna(country):
            country = map_to_country_itl(bplace, itl_cities)

        if not pd.isna(country):
            country = country.strip()
            if country in country_alias_dict:
                country = country_alias_dict[country]
            ret.loc[i] = country
        else:
            ret.loc[i] = pd.NA

        # ret.loc[i] = country if not pd.isna(country) else pd.NA

    return ret


def map_country_to_continent(country_col, country_to_continent):
    ret = pd.Series(index=country_col.index, dtype="string")
    for i, country in country_col.items():
        if country is None or pd.isna(country):
            ret.loc[i] = pd.NA
            continue
        # val = country_to_continent[country]
        val = country_to_continent.get(country)
        if val is None:
            ret.loc[i] = pd.NA
            continue

        if type(val) is str:
            ret.loc[i] = val
        else:
            ret.loc[i] = val[0] + "," + val[1]
    return ret


# def sub_emeriti_costnr(costnr, kostst_emeriti_map):
#     # return modified column and mask where modification was made
#     mask = costnr.isin(kostst_emeriti_map.keys())
#     modified_costnr = costnr.copy()
#     modified_costnr[mask] = costnr[mask].map(lambda x: kostst_emeriti_map[x])
#     return modified_costnr, mask


def sub_emeriti_costnr(costnr, kostst_emeriti_map):
    # normalize input to comparable strings
    # s = (
    #     costnr.astype("string")
    #     .str.strip()
    #     .str.replace(r"\.0$", "", regex=True)  # handles 30001234.0 from Excel
    # )
    s = _canon_ks(costnr)

    emer_map = {str(k).strip(): str(v).strip() for k, v in kostst_emeriti_map.items()}

    mask = s.isin(emer_map)
    modified = costnr.copy()
    modified.loc[mask] = s.loc[mask].map(emer_map)
    return modified, mask


def get_first_valid_starting_with_3(series):
    """Return first cost number starting with '3' and has 8 digits, or None."""
    return next((str(val) for val in series if str(val).startswith("3")), None)


def clean_costnr(
    costnr,
    firstname,
    lastname,
    internal,
    kostst_emeriti_map,
    new_kostst_list,
    kostst_history,
    kostst_history_old,
    use_successor=False,
    verbose=False,
):
    """
    Attempts to impute or correct cost numbers (Kostenstellen) by matching
    name and internal status using various data sources.

    If use_successor=True, will attempt to use Nachfolge-kostenstelle as a fallback.
    """

    def impute_by_name(lname, fname, use_successor):
        """Try multiple lookup tables to find valid kostenstelle based on name."""
        if pd.isna(lname):
            return None

        def by_name_from_newer_table(table, full_name):
            ks = table[
                table["Kostenstellenverantwortliche/r"].str.contains(
                    full_name, na=False, regex=False
                )
                & (table["Kategorie"] == "Fachgebiet")
                & ~table[
                    "Bemerkungen / Hinweise, Information zur Professur"
                ].str.contains("komissarisch", na=False, regex=False, case=False)
            ]["Kostenstelle"]
            return get_first_valid_starting_with_3(ks)

        def get_hist_rows(table, full_name):
            col = (
                "Professor / Kostenstellen-verantwortlicher"
                if "Professor / Kostenstellen-verantwortlicher" in table.columns
                else "Professor/Kostenstellen-verantwortlicher"
            )
            ks = table[
                table[col].str.contains(full_name, na=False, regex=False)
                & ~table["Bemerkungen / Hinweise"].str.contains(
                    "komissarisch", na=False, regex=False, case=False
                )
            ]
            return ks

        # def get_hist_rows_lname_only(table, lname):
        #     col = (
        #         "Professor / Kostenstellen-verantwortlicher"
        #         if "Professor / Kostenstellen-verantwortlicher" in table.columns
        #         else "Professor/Kostenstellen-verantwortlicher"
        #     )
        #     ks = table[
        #         table[col].str.contains(
        #             re.escape(lname) + r"(?:,|$)", na=False, regex=True
        #         )
        #         & ~table["Bemerkungen / Hinweise"].str.contains(
        #             "komissarisch", na=False, regex=False
        #         )
        #     ]
        #     return ks
        def get_hist_rows_lname_only(table, lname):
            col = (
                "Professor / Kostenstellen-verantwortlicher"
                if "Professor / Kostenstellen-verantwortlicher" in table.columns
                else "Professor/Kostenstellen-verantwortlicher"
            )

            lname = lname.strip()

            lastname_field = (
                table[col]
                .astype("string")
                .str.strip()
                .str.split(",", n=1)
                .str[0]  # before comma OR full string if no comma
                .str.strip()
            )

            mask_name = lastname_field.eq(lname)

            mask_not_kommissarisch = ~table["Bemerkungen / Hinweise"].astype(
                "string"
            ).str.contains("komissarisch", na=False, regex=False)
            return table[mask_name & mask_not_kommissarisch]

        def by_name_from_older_table(table, full_name):
            ks = get_hist_rows(table, full_name)
            return get_first_valid_starting_with_3(ks["Kosten-stelle"])

        def by_lastname_from_older_table(table, lname):
            ks = get_hist_rows_lname_only(table, lname)
            return get_first_valid_starting_with_3(ks["Kosten-stelle"])

        fname = "" if pd.isna(fname) else fname
        full_name = f"{lname}, {fname}"

        # try newest lists
        for table in new_kostst_list:
            replacement = by_name_from_newer_table(table, full_name)
            if replacement:
                return replacement

        # Try 2: kostst_history (full name)
        replacement = by_name_from_older_table(kostst_history, full_name)
        if replacement:
            return replacement

        # Try 3: kostst_history_old (fullname)
        replacement = by_name_from_older_table(kostst_history_old, full_name)
        if replacement:
            return replacement

        # Try 4: kostst_history_old (lastname only, last resort!). This leads
        # to false positives, so we are commenting this out
        # replacement = by_lastname_from_older_table(kostst_history_old, lname)
        # if replacement:
        #     return replacement

        # Successor columns if allowed
        if use_successor:
            hist_rows = get_hist_rows(kostst_history, full_name)
            if "Nachfolge-kosten-stelle" in hist_rows.columns:
                replacement = get_first_valid_starting_with_3(
                    hist_rows["Nachfolge-kosten-stelle"]
                )
                if replacement:
                    return replacement

            hist_old_rows = get_hist_rows(kostst_history_old, full_name)
            if "Nachfolge-kostenstelle" in hist_old_rows.columns:
                replacement = get_first_valid_starting_with_3(
                    hist_old_rows["Nachfolge-kostenstelle"]
                )
                if replacement:
                    return replacement

            # remove only by lname due to false positives
            # hist_old_rows_lname = get_hist_rows_lname_only(kostst_history_old, lname)
            # if "Nachfolge-kostenstelle" in hist_old_rows_lname.columns:
            #     replacement = get_first_valid_starting_with_3(
            #         hist_old_rows_lname["Nachfolge-kostenstelle"]
            #     )
            #     if replacement:
            #         return replacement

        return None

    # Step 1: Handle emeriti overrides

    modified_costnr, no_touch_mask = sub_emeriti_costnr(costnr, kostst_emeriti_map)
    no_touch_mask = pd.Series(no_touch_mask, index=modified_costnr.index).astype(bool)

    # Step 2: Find invalid entries
    matches_costnr_does_not_exist, _ = check_kostenstelle(
        modified_costnr, new_kostst_list, kostst_history, kostst_history_old
    )

    # matches_costnr_does_not_exist[no_touch_mask] = False
    matches_costnr_does_not_exist.loc[no_touch_mask] = False

    # Step 3: Prepare iterables
    faulty_costnr = modified_costnr[matches_costnr_does_not_exist]
    faulty_firstname = firstname[matches_costnr_does_not_exist]
    faulty_lastname = lastname[matches_costnr_does_not_exist]
    faulty_internal = internal[matches_costnr_does_not_exist]

    for i, (fname, lname, is_internal) in enumerate(
        zip(faulty_firstname, faulty_lastname, faulty_internal)
    ):
        idx = faulty_costnr.index[i]
        replacement = impute_by_name(lname, fname, use_successor)

        if replacement:
            modified_costnr.loc[idx] = replacement
            no_touch_mask.loc[idx] = True
            if verbose:
                print(f"{faulty_costnr.iloc[i]} → {replacement}")
            continue
        else:
            modified_costnr.loc[idx] = pd.NA
            if verbose:
                print(f"{faulty_costnr.iloc[i]} → NaN, no replacement found")

    matches_costnr_name_nofit, n_mismatch = crosscheck_kostenstelle_name(
        modified_costnr,
        firstname,
        lastname,
        internal,
        new_kostst_list,
        kostst_history,
        kostst_history_old,
    )
    matches_costnr_name_nofit.loc[no_touch_mask] = False

    # leftover_short_mask = (
    #     ~no_touch_mask
    #     & ~(matches_costnr_does_not_exist | matches_costnr_name_nofit)
    #     & (costnr.str.len() < 4)
    # )

    # for each match, try to find replacement by name. If name exists, replace, else if number exists, keep it, otherwise set to NaN
    faulty_costnr = modified_costnr[matches_costnr_name_nofit]
    faulty_firstname = firstname[matches_costnr_name_nofit]
    faulty_lastname = lastname[matches_costnr_name_nofit]
    faulty_internal = internal[matches_costnr_name_nofit]
    for i, (fname, lname, is_internal, cnum) in enumerate(
        zip(faulty_firstname, faulty_lastname, faulty_internal, faulty_costnr)
    ):
        idx = faulty_costnr.index[i]
        replacement = impute_by_name(lname, fname, use_successor)

        if replacement:
            modified_costnr.loc[idx] = replacement
            no_touch_mask.loc[idx] = True
            if verbose:
                print(f"{cnum} → {replacement}, {lname}, crosschecked by name")
        else:
            # If no replacement found, keep the original number
            cnum_str = str(cnum) if pd.notna(cnum) else None
            if cnum_str and len(cnum_str) >= 4:
                modified_costnr.loc[idx] = cnum

    return modified_costnr


def num_to_bool(col):
    return col == 1


def check_nonempty(*args):
    ret = args[0].notna()
    for arg in args[1:]:
        ret = ret | arg.notna()
    return ret


def get_fullname(firstname, lastname):
    fullname = firstname + " " + lastname
    fullname[lastname.isna()] = pd.NA
    return fullname


def defense_to_bool(col):
    def defense_map(entry):
        match entry:
            case "Bestanden":
                return True
            case "Nicht Bestanden":
                return False
            case _:
                return pd.NA

    return col.apply(defense_map).astype("boolean")


def grade_to_float(grade_col):
    def grademap(grade):
        if pd.isna(grade):
            return pd.NA
        match grade:
            case "Mit Auszeichnung":
                return 0.0
            case "Sehr gut":
                return 1.0
            case "Gut":
                return 2.0
            case "Befriedigend":
                return 3.0
            case "Ausreichend":
                return 4.0
            case "Bestanden, Gesamtnote nicht bekannt":
                return "passed, no grade"
            case "Bestanden":
                return "passed, no grade"
            case _:
                return pd.NA

    return grade_col.apply(grademap)


def noacc_noabrt_nodef_check(acc_date, abort, def_date):
    noacc_mask = acc_date.isna()
    noabrt_mask = abort.isna() | ~abort
    nodef_mask = def_date.isna()
    matches = noacc_mask & noabrt_mask & nodef_mask
    return ~matches


def sem_to_bool(sem_col):
    nan_mask = sem_col.isna()
    ret = sem_col == "WINTER"
    ret = ret.astype("boolean")
    ret[nan_mask] = pd.NA
    return ret


def extract_uni(uni, uni2, state):
    mask_uni, _ = check_uni(uni, uni2)
    mask_unistate, _ = check_unistaat(uni, state)
    mask_invalid = mask_uni | mask_unistate

    mask_german_uni = uni != "Hochschule im Ausland"
    mask_foreign_uni = ~mask_german_uni

    ret = pd.Series(index=uni.index, dtype="string")

    ret[mask_german_uni] = uni[mask_german_uni]
    ret[mask_foreign_uni] = uni2[mask_foreign_uni]
    ret[mask_invalid] = pd.NA
    return ret


def uni_is_german(uni, uni2, state):
    mask_uni, _ = check_uni(uni, uni2)
    mask_unistate, _ = check_unistaat(uni, state)
    mask_invalid = mask_uni | mask_unistate

    mask_german_uni = uni != "Hochschule im Ausland"

    ret = mask_german_uni.astype("boolean")
    ret[mask_invalid] = pd.NA

    return ret


def yn_to_bool(col, na=None):
    assert na is None or na == "Ja" or na == "Nein"
    if na is not None:
        return col.fillna(na) == "Ja"
    return col == "Ja"


def clean_year(col):
    col = col.copy()
    invalid_mask = ~col.apply(
        lambda x: (isinstance(x, int) or isinstance(x, float)) and 1900 <= x <= 2100
    )
    col.loc[invalid_mask] = pd.NA
    return col.astype("Int64")  # Use Int64 to allow NA values


def check_german_cship(citizenship, citizenship2):
    ret = (citizenship == "Deutschland") | (citizenship2 == "Deutschland")
    ret = ret.astype("boolean")
    both_na = citizenship.isna() & citizenship2.isna()
    ret[both_na] = pd.NA
    return ret


def copy_col(col, na=None):
    if na is not None:
        return col.fillna(na)
    return col.copy()


def copy_and_compare(col, ref_col):
    print(col.name, col.dtype, ref_col.dtype)
    col_orig = col.copy()  # untouched original
    ref_col = ref_col.reindex(col_orig.index)

    a = col_orig
    b = ref_col

    # 3) Booleans: normalize common encodings ("Ja"/"Nein", 1/0, True/False)
    if is_bool_dtype(a) or is_bool_dtype(b):
        print(col, ref_col)

        def to_bool(s):
            if is_bool_dtype(s):
                return s.astype("boolean")
            s2 = s.astype("string").str.strip().str.lower()
            out = pd.Series(pd.NA, index=s.index, dtype="boolean")
            out[s2.isin(["true", "1", "ja", "yes"])] = True
            out[s2.isin(["false", "0", "nein", "no"])] = False
            return out

        a_bool = to_bool(a)
        b_bool = to_bool(b)
        eq = a_bool.eq(b_bool)

        check = (eq.fillna(False)) | (a_bool.isna() & b_bool.isna())
        return col_orig, check.astype("boolean")

    # 1) Numeric: compare numerically (so 1 == 1.0, "1" == 1, etc.)
    if is_numeric_dtype(a) or is_numeric_dtype(b):
        a_num = pd.to_numeric(a, errors="coerce")
        b_num = pd.to_numeric(b, errors="coerce")
        eq = a_num.eq(b_num)

        check = (eq.fillna(False)) | (a_num.isna() & b_num.isna())
        return col_orig, check.astype("boolean")

    # 2) Datetime: compare after parsing (optional: set your format if strict)
    if is_datetime64_any_dtype(a) or is_datetime64_any_dtype(b):
        a_dt = pd.to_datetime(a, errors="coerce", dayfirst=True)
        b_dt = pd.to_datetime(b, errors="coerce", dayfirst=True)
        eq = a_dt.eq(b_dt)

        check = (eq.fillna(False)) | (a_dt.isna() & b_dt.isna())
        return col_orig, check.astype("boolean")

    # 4) Fallback: string compare but without numeric artifacts,
    #    because numeric handled above.
    a_str = a.astype("string").str.strip()
    b_str = b.astype("string").str.strip()
    eq = a_str.eq(b_str)

    check = (eq.fillna(False)) | (a_str.isna() & b_str.isna())
    return col_orig, check.astype("boolean")


def check_col_intergrity(df, diagnostics=True, diagnostics_full=False):
    cols = get_cols(df)
    (
        faculties,
        genders,
        citizenships,
        pagin_nr,
        first_name,
        last_name,
        bdate,
        reg_stop,
        tu_cert,
        imma,
        hzb_type,
        year_first_hzb,
        hzb_state,
        hzb_bl,
        erstimma_staat,
        erstimma_uni,
        erstimma_uni2,
        erstimma_sem,
        erstimma_yr,
        pba_passed,
        pba_state,
        pba_uni,
        pba_uni2,
        pba_type,
        pba_subject,
        pba_date,
        pba_grade,
        pba_degree,
        phd_prog,
        phd_subject,
        coop_name,
        pub_form,
        type_coop,
        accept_date,
        report_yr_reg,
        promreg,
        defense_date,
        result,
        abort,
        abort_date,
        advisor_firstname,
        advisor_lastname,
        advisor_costnr,
        ref1_type,
        ref1_internal,
        ref1_firstname,
        ref1_lastname,
        ref1_costnr,
        ref2_type,
        ref2_internal,
        ref2_firstname,
        ref2_lastname,
        ref2_costnr,
        ref3_internal,
        ref3_lastname,
        ref3_costnr,
        citizenship2,
        place_of_birth,
        tubesch_nr,
        matrnr_student,
        matrnr_prom,
        type_finance,
        finance_start,
        fiance_end,
        process,
        program_name,
        program_start,
        program_end,
        prom_degree,
        aufl1,
        aufl3,
        title,
        timetable,
        abroad_country,
        abroad_type,
        abroad_from,
        abroad_to,
        mobility_program,
        break_from,
        break_to,
        chair_firstname,
        chair_lastname,
        chair_costnr,
        ref1_lecode,
        # advisor_addinfo,
        ref2_secretary,
        ref2_lecode,
        ref3_type,
        ref3_firstname,
        ref3_secretary,
        ref3_lecode,
        open_date,
        open_date_application,
        review_deadline,
        title_de,
        title_en,
        dis_lng,
        grade_latin,
        grade_name,
        date_publication,
        pubnr,
        orcid,
        abstract,
        ub_abstract_date,
        certificate_date,
        notes,
        archive_nr,
        linf_planned_date,
        linf_exp_date,
        connected,
        dbaseid,
        stala_id,
        stala_pseud,
        worked_on,
        complete,
        checked,
        email_exists,
        yr_defense,
        month_defense,
        is_german,
    ) = cols

    issues_table = pd.DataFrame(columns=["col_name", "n_issues", "n_unique_vals"])
    issue_dict = {}
    vals_dict = {}
    mask_table = pd.DataFrame(dtype=bool)

    checks_dict = {
        "Paginiernr": ((pagin_nr,), {}, check_positive_number, [], None),
        "Fakultaet": (
            (faculties, r"^Fakultät (?:I|II|III|IV|V|VI|VII)$"),
            {},
            check_permitted_regex,
            [],
            None,
        ),
        "Name": ((first_name, last_name), {}, check_names, "Nachname", None),
        "Geschlecht": ((genders, r"m|w|d"), {}, check_permitted_regex, [], None),
        "Gebdat": ((bdate,), {}, check_date, [], None),
        "Staatsang": (
            (citizenships, country_names_ger),
            {},
            check_in_permitted_list,
            [],
            None,
        ),
        "Abbruch": ((abort, r"Ja|Nein"), {}, check_permitted_regex, [], None),
        "Aussprache_Dat": ((defense_date,), {"na": True}, check_date, [], None),
        "Registrierungsstop": (
            (reg_stop, abort, defense_date),
            {},
            check_regstop,
            ["Registrierungsstop", "Aussprache_Dat", "Abbruch"],
            "Registrierungsstop",
        ),
        "TUBesch": ((tu_cert, r"Ja|Nein"), {}, check_permitted_regex, [], None),
        "Imma": ((imma, r"Ja|Nein"), {}, check_permitted_regex, [], None),
        "HZB_Art": ((hzb_type, hzb_types), {}, check_in_permitted_list, [], None),
        "JahrersteHochschulzugangsberec": (
            (year_first_hzb,),
            {},
            check_valid_year,
            [],
            None,
        ),
        "HZB_Staat": (
            (hzb_state, country_names_ger),
            {},
            check_in_permitted_list,
            [],
            None,
        ),
        "HZB_Land/BL": (
            (hzb_bl, hzb_state),
            {},
            check_hzb_bl,
            ["HZB_Staat", "HZB_BL"],
            "HZB_BL",
        ),
        "Erstimma_Staat": (
            (erstimma_staat, country_names_ger),
            {},
            check_in_permitted_list,
            [],
            None,
        ),
        "firstimma_abroad_uni": (
            (erstimma_uni, erstimma_staat),
            {},
            check_unistaat,
            ["Erstimma_Staat", "Erstimma_Hochschule", "Erstimma_Hochschule2"],
            "Erstimma_Hochschule",
        ),
        "Erstimma_Hochschule": (
            (erstimma_uni, erstimma_uni2),
            {},
            check_uni,
            ["Erstimma_Hochschule", "Erstimma_Hochschule2"],
            "Erstimma_Hochschule",
        ),
        "Erstimma_Sem": (
            (erstimma_sem, r"WINTER|SOMMER"),
            {},
            check_permitted_regex,
            [],
            None,
        ),
        "Erstimma_Jahr": ((erstimma_yr,), {}, check_valid_year, [], None),
        "pba_Bestanden": (
            (pba_passed, r"Ja|Nein"),
            {},
            check_permitted_regex,
            [],
            None,
        ),
        "pba_Staat": (
            (pba_state, country_names_ger),
            {},
            check_in_permitted_list,
            [],
            None,
        ),
        "pba_unistate": (
            (pba_uni, pba_state),
            {},
            check_unistaat,
            ["pba_Staat", "pba_Hochschule", "pba_Hochschule2"],
            "pba_Hochschule",
        ),
        "pba_Art": ((pba_type, pba_types), {}, check_in_permitted_list, [], None),
        "pba_Hochschule": (
            (pba_uni, pba_uni2),
            {},
            check_uni,
            ["pba_Hochschule", "pba_Hochschule2"],
            "pba_Hochschule",
        ),
        "pba_Fach": (
            (pba_subject, pba_subjects),
            {},
            check_in_permitted_list,
            [],
            None,
        ),
        "pba_Dat": ((pba_date,), {}, check_date, [], None),
        "pba_Note": ((pba_grade, grades), {}, check_in_permitted_list, [], None),
        "PromProgramm": ((phd_prog, r"Ja|Nein"), {}, check_permitted_regex, [], None),
        "Promfach": (
            (phd_subject, phd_subjects),
            {},
            check_in_permitted_list,
            [],
            None,
        ),
        "Koop_Name": (
            (coop_name, coops),
            {"na": True},
            check_in_permitted_list,
            [],
            None,
        ),
        "Art_Form": ((pub_form, pub_forms), {}, check_in_permitted_list, [], None),
        "Art_Koop": ((type_coop, coop_types), {}, check_in_permitted_list, [], None),
        "Annahme_Dat": ((accept_date,), {}, check_date, [], None),
        "Berichtsjahr_reg": (
            (report_yr_reg,),
            {"na": True},
            check_valid_year,
            [],
            None,
        ),
        "Promreg": (
            (promreg, promregs),
            {"na": True},
            check_in_permitted_list,
            [],
            None,
        ),
        "Ergebnis": (
            (result, r"Bestanden|Nicht Bestanden"),
            {},
            check_permitted_regex,
            [],
            None,
        ),
        "Abbruch_Dat": ((abort_date,), {"na": True}, check_date, [], None),
        "Betr_Name": (
            (advisor_firstname, advisor_lastname),
            {},
            check_names,
            ["Betr_Vorname", "Betr_Name"],
            "Betr_Name",
        ),
        "Betr_Kostst": ((advisor_costnr,), {}, check_kostenstelle, [], None),
        "Betr_Kostst_Name": (
            (
                advisor_costnr,
                advisor_firstname,
                advisor_lastname,
                pd.Series(["Ja"] * len(df)),
            ),
            {},
            crosscheck_kostenstelle_name,
            ["Betr_Vorname", "Betr_Name", "Betr_Kostst"],
            "Betr_Kostst",
        ),
        "Gutacht1_Art": (
            (ref1_type, r"Erstgutachter"),
            {},
            check_permitted_regex,
            [],
            None,
        ),
        "Gutacht1_Intern": (
            (ref1_internal, r"Ja|Nein"),
            {},
            check_permitted_regex,
            [],
            None,
        ),
        "Gutacht1_Name": (
            (ref1_firstname, ref1_lastname),
            {},
            check_names,
            ["Gutacht1_VorName", "Gutacht1_Name"],
            "Gutacht1_Name",
        ),
        "Gutacht1_Kostst": ((ref1_costnr,), {}, check_kostenstelle, [], None),
        "Gutacht1_Kostst_Name": (
            (ref1_costnr, ref1_firstname, ref1_lastname, ref1_internal),
            {},
            crosscheck_kostenstelle_name,
            ["Gutacht1_VorName", "Gutacht1_Name", "Gutacht1_Kostst"],
            "Gutacht1_Kostst",
        ),
        "Gutacht2_Art": (
            (ref2_type, r"Zweitgutachter"),
            {},
            check_permitted_regex,
            [],
            None,
        ),
        "Gutacht2_Intern": (
            (ref2_internal, r"Ja|Nein"),
            {},
            check_permitted_regex,
            [],
            None,
        ),
        "Gutacht2_Name": (
            (ref2_firstname, ref2_lastname),
            {},
            check_names,
            ["Gutacht2_VorName", "Gutacht2_Name"],
            "Gutacht2_Name",
        ),
        "Gutacht2_Kostst": ((ref2_costnr,), {}, check_kostenstelle, [], None),
        "Gutacht2_Kostst_Name": (
            (ref2_costnr, ref2_firstname, ref2_lastname, ref2_internal),
            {},
            crosscheck_kostenstelle_name,
            ["Gutacht2_VorName", "Gutacht2_Name", "Gutacht2_Kostst"],
            "Gutacht2_Kostst",
        ),
        "Gutacht3_Intern": (
            (ref3_internal, r"Ja|Nein"),
            {},
            check_permitted_regex,
            [],
            None,
        ),
        "Gutacht3_Name": (
            (ref3_firstname, ref3_lastname),
            {},
            check_names,
            ["Gutacht3_VorName", "Gutacht3_Name"],
            "Gutacht3_Name",
        ),
        "Gutacht3_Kostst": ((ref3_costnr,), {}, check_kostenstelle, [], None),
        "Gutacht3_Kostst_Name": (
            (ref3_costnr, ref3_firstname, ref3_lastname, ref3_internal),
            {},
            crosscheck_kostenstelle_name,
            ["Gutacht3_VorName", "Gutacht3_Name", "Gutacht3_Kostst"],
            "Gutacht3_Kostst",
        ),
        "Staatsang2": (
            (citizenship2, country_names_ger),
            {"na": True},
            check_in_permitted_list,
            [],
            None,
        ),
        "Gebort": ((place_of_birth,), {}, check_birthplace, [], None),
        "Art_Finanz": (
            (
                type_finance,
                [
                    "Finanzierung aus Beschäftigung in einer außeruniversitären Forschungseinrichtung",
                    "Finanzierung als Drittmittelbeschäftigte/r",
                    "Finanzierung als Haushaltsbeschäftigte/r",
                    "Stipendium",
                ],
            ),
            {"na": True},
            check_in_permitted_list,
            [],
            None,
        ),
        "Finanz_Start": ((finance_start,), {"na": True}, check_date, [], None),
        "Finanz_End": ((fiance_end,), {"na": True}, check_date, [], None),
        "Verfahren": ((process, r"Ja|Nein"), {}, check_permitted_regex, [], None),
        "PromGrad": (
            (
                prom_degree,
                [
                    "Dr.-Ing.",
                    "Dr. rer. nat.",
                    "Dr. phil.",
                    "Dr. rer. oec.",
                    "Dr. P.H.",
                    "Dr. sc. agr.",
                ],
            ),
            {},
            check_in_permitted_list,
            [],
            None,
        ),
        "Auflage1": ((aufl1, r"Ja|Nein"), {}, check_permitted_regex, [], None),
        "Auflage3": ((aufl3, r"Ja|Nein"), {}, check_permitted_regex, [], None),
        "Vorsitz_Name": (
            (chair_firstname, chair_lastname),
            {},
            check_names,
            ["VorsitzenderVorname", "VorsitzenderNachname"],
            "VorsitzenderNachname",
        ),
        "Vorsitz_Kostst": ((chair_costnr,), {}, check_kostenstelle, [], None),
        "Vorsitz_Kostst_Name": (
            (
                chair_costnr,
                chair_firstname,
                chair_lastname,
                pd.Series(["Ja"] * len(df)),
            ),
            {},
            crosscheck_kostenstelle_name,
            ["VorsitzenderVorname", "VorsitzenderNachname", "Vorsitz_Kostst"],
            "Vorsitz_Kostst",
        ),
        "Gutacht3_Art": (
            (ref3_type, r"Drittgutachter"),
            {},
            check_permitted_regex,
            [],
            None,
        ),
        "Eröff_Dat": ((open_date,), {"na": True}, check_date, [], None),
        "Eröff_Dat_Antrag": (
            (open_date_application,),
            {"na": True},
            check_date,
            [],
            None,
        ),
        "Gutachtenfrist": ((review_deadline,), {"na": True}, check_date, [], None),
        "Diss_Sparche": (
            (dis_lng, r"de|en"),
            {},
            check_permitted_regex,
            [],
            None,
        ),
        "Note_Latein": (
            (
                grade_latin,
                [
                    "Summa cum laude",
                    "Magma cum laude",
                    "Cum laude",
                ],
            ),
            {},
            check_in_permitted_list,
            [],
            None,
        ),
        "Note_Name": (
            (
                grade_name,
                [
                    "Mit Auszeichnung",
                    "Sehr gut",
                    "Gut",
                    "Bestanden",
                    "Bestanden, Gesamtnote nicht bekannt",
                ],
            ),
            {},
            check_in_permitted_list,
            [],
            None,
        ),
        "Note": (
            (grade_latin, grade_name),
            {},
            crosscheck_grade,
            ["Note_Latein", "Note_Name"],
            "Note_Latein",
        ),
        "Publikationsfr": ((date_publication,), {"na": True}, check_date, [], None),
        "UbAbstrakt": ((ub_abstract_date,), {}, check_date, [], None),
        "Urkune_Dat": ((certificate_date,), {}, check_date, [], None),
        "Linf_planned": ((linf_planned_date,), {}, check_date, [], None),
        "Linf_exp": ((linf_exp_date,), {}, check_date, [], None),
        "Verknüft": ((connected, r"Ja|Nein"), {}, check_permitted_regex, [], None),
        "bearb": ((worked_on,), {}, check_date, [], None),
        "vollständig": ((complete, r"Ja|Nein"), {}, check_permitted_regex, [], None),
        "kontrolliert": ((checked, r"Ja|Nein"), {}, check_permitted_regex, [], None),
        "Mail_vorh": ((email_exists,), {}, check_boolint, [], None),
        "Jahr_Aussprache": ((yr_defense,), {}, check_valid_year, [], None),
        "Mon_Aussprache": ((month_defense,), {}, check_positive_number, [], None),
        "Deutsch": ((is_german,), {}, check_boolint, [], None),
    }

    test_keys = list(checks_dict.keys())
    for field_name in test_keys:
        args, kwargs, func, field_name_native, unique_by_field = checks_dict[field_name]
        matches, n_issues = func(*args, **kwargs)
        issues_table = commit_issue_data(
            df,
            issues_table,
            issue_dict,
            vals_dict,
            field_name,
            n_issues,
            matches,
            mask_table,
            field_name_native=field_name_native,
            unique_by_field=unique_by_field,
        )

    return vals_dict, issues_table, issue_dict, mask_table

    citizenships_alias = alias_country_names(citizenships)
    # if Staatsand_alias exists, write citizenships to df["Staatsang"], else: create it
    df["Staatsang_alias"] = citizenships_alias
    # move later

    hzb_state_alias = alias_country_names(hzb_state)
    df["HZB_Staat_alias"] = hzb_state_alias


def check_positive_number(input):
    matches = ~input.apply(
        lambda x: (isinstance(x, int) or isinstance(x, float)) and x >= 0
    )
    n_issues = sum(matches)
    return matches, n_issues


def check_permitted_regex(input, permitted_pattern):
    matches = ~input.str.match(permitted_pattern, na=False)
    n_issues = sum(matches)
    return matches, n_issues


def check_names(first_name, last_name):
    # check if either is empty, i.e. either nan or space
    first_name_nan_mask = first_name.isna() | (first_name.str.strip() == "")
    last_name_nan_mask = last_name.isna() | (last_name.str.strip() == "")
    problem_mask = first_name_nan_mask | last_name_nan_mask
    return problem_mask, sum(problem_mask)


def check_boolint(input):
    matches = ~((input == 0) | (input == 1))
    n_issues = sum(matches)
    return matches, n_issues


def check_date(input, na=False):
    # Check if the date is in the format YYYY-MM-DD
    pattern = r"^\d{2}-\d{2}-\d{4}$"
    matches = ~input.str.match(pattern, na=na)
    n_issues = sum(matches)
    return matches, n_issues


def check_in_permitted_list(input, permitted_values, na=False):
    inp_strpped = input.astype("string").str.strip()
    if not na:
        nan_factor = True
    else:
        nan_factor = ~input.isna()

    matches = ~inp_strpped.isin(permitted_values) & nan_factor
    n_issues = sum(matches)
    return matches, n_issues


def alias_country_names(col, country_alias_dict):
    """
    Replace country names in a pandas Series according to a provided alias dictionary.
    Parameters:
    -col: pandas Series containing country names to be aliased.
    -country_alias_dict: Dictionary mapping original country names to their aliases.
    Returns:
    Aliased pandas Series with country names replaced according to the alias dictionary.
    """
    s = col.astype("string").str.strip()
    return s.replace(country_alias_dict)  # dict replacement is vectorized and exact


def check_regstop(reg_stop, abort, defense_date):
    pattern = r"Ja|Nein"
    matches_regstop = ~reg_stop.str.match(pattern, na=False)
    abort_mask = abort.map(lambda x: x == "Ja")
    defense_mask = pd.isna(defense_date)
    matches = matches_regstop & (abort_mask | ~defense_mask)
    n_issues = sum(matches)

    return matches, n_issues


def check_valid_year(input, na=False):
    if not na:
        nan_factor = True
    else:
        nan_factor = ~input.isna()
    matches = (
        ~input.apply(
            lambda x: (isinstance(x, int) or isinstance(x, float)) and 1900 <= x <= 2100
        )
        & nan_factor
    )
    n_issues = sum(matches)
    return matches, n_issues


def check_hzb_bl(hzb_bl, hzb_state):
    hzb_bl_stripped = hzb_bl.str.strip()
    hzb_states_stripped = hzb_state.str.strip()
    matches = ~hzb_bl_stripped.isin(bl_names) & hzb_states_stripped.isin(
        ["Deutschland"]
    )
    n_issues = sum(matches)
    return matches, n_issues


def check_uni(uni1, uni2):
    matches1 = uni1.notna() & (uni1.str.strip() != "")
    matches2 = (
        uni1.isin(["Hochschule im Ausland"]) & uni2.notna() & (uni2.str.strip() != "")
    )
    matches = ~(matches1 | matches2)
    n_issues = sum(matches)
    return matches, n_issues


def check_unistaat(uni, staat):
    matches = ~(
        (
            (staat.str.strip() != "Deutschland")
            & (uni.str.strip() == "Hochschule im Ausland")
        )
        | (
            (uni.str.strip() != "Hochschule im Ausland")
            & (staat.str.strip() == "Deutschland")
        )
    )
    n_issues = sum(matches)
    return matches, n_issues


def crosscheck_kostenstelle_name(
    kostenstelle,
    firstnames,
    lastnames,
    internal,
    new_kostst_list,
    kostst_history,
    kostst_history_old,
):
    def get_all_ks_numbers_from_new_table(table):
        ks_numbers = table[
            table["Kostenstellenverantwortliche/r"].str.contains(
                namestring, na=False, regex=False
            )
            & (table["Kategorie"] == "Fachgebiet")
        ]["Kostenstelle"]
        return ks_numbers

    # also: check if name -> number in any of the three tables
    matches = pd.Series([False] * len(kostenstelle), dtype=bool)
    for i, (firstname, lastname, ks, is_internal) in enumerate(
        zip(firstnames, lastnames, kostenstelle, internal)
    ):
        is_internal = is_internal == "Ja"
        if pd.isna(firstname):
            firstname = ""
        if pd.isna(lastname) or pd.isna(ks):
            continue
        namestring = lastname + ", " + firstname

        newer_numbers_list = [
            get_all_ks_numbers_from_new_table(table) for table in new_kostst_list
        ]

        ks_new_numbers = kostst_history[
            kostst_history["Professor / Kostenstellen-verantwortlicher"].str.contains(
                namestring, na=False, regex=False
            )
        ]["Kosten-stelle"]
        ks_old_numbers = kostst_history_old[
            kostst_history_old["Professor/Kostenstellen-verantwortlicher"].str.contains(
                lastname, na=False, regex=False
            )
        ]["Kosten-stelle"]

        list_of_ks_series = newer_numbers_list + [ks_new_numbers, ks_old_numbers]
        ok = any(ks in list(ks_list) for ks_list in list_of_ks_series)

        matches.iloc[i] = (not ok) and (is_internal)

    n_issues = sum(matches)
    return matches, n_issues


def crosscheck_kostenstelle_name(
    kostenstelle,
    firstnames,
    lastnames,
    internal,
    new_kostst_list,
    kostst_history,
    kostst_history_old,
):
    def get_all_ks_numbers_from_new_table(table, namestring):
        return table[
            table["Kostenstellenverantwortliche/r"].str.contains(
                namestring, na=False, regex=False
            )
            & (table["Kategorie"] == "Fachgebiet")
        ]["Kostenstelle"]

    matches = pd.Series(False, index=kostenstelle.index, dtype=bool)

    # precompute column names robustly (your history uses different spellings)
    col_hist = (
        "Professor / Kostenstellen-verantwortlicher"
        if "Professor / Kostenstellen-verantwortlicher" in kostst_history.columns
        else "Professor/Kostenstellen-verantwortlicher"
    )
    col_old = (
        "Professor / Kostenstellen-verantwortlicher"
        if "Professor / Kostenstellen-verantwortlicher" in kostst_history_old.columns
        else "Professor/Kostenstellen-verantwortlicher"
    )

    for i, (firstname, lastname, ks, is_internal) in enumerate(
        zip(firstnames, lastnames, kostenstelle, internal)
    ):
        if pd.isna(lastname) or pd.isna(ks):
            continue

        is_internal = is_internal == "Ja"
        firstname = "" if pd.isna(firstname) else firstname
        namestring = f"{lastname}, {firstname}"

        ks_c = _canon_ks_one(ks)
        if ks_c is None:
            continue

        # collect all candidate numbers and canonicalize
        candidates = []

        for table in new_kostst_list:
            candidates.append(get_all_ks_numbers_from_new_table(table, namestring))

        candidates.append(
            kostst_history[
                kostst_history[col_hist].str.contains(namestring, na=False, regex=False)
            ]["Kosten-stelle"]
        )

        candidates.append(
            kostst_history_old[
                kostst_history_old[col_old].str.contains(
                    namestring, na=False, regex=False
                )
            ]["Kosten-stelle"]
        )

        # flatten to a set of canonical strings
        cand_set = {
            _canon_ks_one(x) for ser in candidates for x in ser.dropna().tolist()
        }
        cand_set.discard(None)

        ok = ks_c in cand_set
        matches.iloc[i] = (not ok) and is_internal

    return matches, int(matches.sum())


def get_unique_ks_from_table(kostst_table):
    ks_mask = (
        (kostst_table["Kategorie"] == "Fachgebiet")
        | (kostst_table["Kategorie"] == "Institut")
        | (kostst_table["Kategorie"] == "Zentrum")
    ) & (kostst_table["Kostenstelle"].str.len() >= 4)
    return kostst_table["Kostenstelle"][ks_mask].unique()


def _canon_ks(x):
    s = pd.Series(x, dtype="string")
    return (
        s.str.strip().str.replace(
            r"\.0$", "", regex=True
        )  # handles Excel floats like 30001234.0
    )


def _canon_ks_one(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    # normalize Excel floats like 30001234.0
    s = re.sub(r"\.0$", "", s)
    return s


# def check_kostenstelle(
#     kostenstelle, new_kostst_list, kostst_history, kostst_history_old
# ):
#     ks_new = kostst_history["Kosten-stelle"].unique()
#     ks_old = kostst_history_old["Kosten-stelle"].unique()
#
#     unique_ks_list = [
#         get_unique_ks_from_table(kostst_table) for kostst_table in new_kostst_list
#     ] + [ks_new, ks_old]
#
#     # number not is individual table
#     matches_by_list = [~kostenstelle.isin(ks) for ks in unique_ks_list]
#     # total matches: number not in any table
#     matches = pd.concat(matches_by_list, axis=1).all(axis=1)
#
#     n_issues = sum(matches)
#     return matches, n_issues
#


def check_kostenstelle(
    kostenstelle, new_kostst_list, kostst_history, kostst_history_old
):
    # canonicalize the input column once
    ks_in = _canon_ks(kostenstelle)

    # canonicalize reference sets
    ks_hist = set(_canon_ks(kostst_history["Kosten-stelle"]).dropna().tolist())
    ks_hist_old = set(_canon_ks(kostst_history_old["Kosten-stelle"]).dropna().tolist())

    unique_ks_list = []
    for kostst_table in new_kostst_list:
        unique_ks_list.append(
            set(_canon_ks(get_unique_ks_from_table(kostst_table)).dropna().tolist())
        )
    unique_ks_list += [ks_hist, ks_hist_old]

    # "does not exist anywhere"
    matches_by_list = [~ks_in.isin(ks_set) for ks_set in unique_ks_list]
    matches = pd.concat(matches_by_list, axis=1).all(axis=1)

    return matches, int(matches.sum())


# def check_pba_degree(pba_degree):
# use dictionary to compile matches/issues
# level_dict = degree_dict["level"]
# discipline_dict = degree_dict["discipline"]


# for level_matches, check if each entry of pba_degree
def check_ger_cities(place_name):
    if isinstance(place_name, str):
        return any(city in place_name for city in german_cities)
    return False


def check_ger_countries(place_name):
    if isinstance(place_name, str):
        return any(country in place_name for country in country_names_ger)
    return False


def map_city_to_country_ger(place_name, german_cities):
    if isinstance(place_name, str):
        for city in german_cities:
            if city in place_name:
                return "Deutschland"
        return pd.NA
    else:
        return pd.NA


def map_to_country_ger(place_name, country_names_ger):
    if isinstance(place_name, str):
        for country in country_names_ger:
            if country in place_name:
                return country
        return pd.NA
    else:
        return pd.NA


def map_to_country_itl(place_name, itl_cities):
    cities_list = itl_cities["city_name_de"]
    if isinstance(place_name, str):
        for i, city in enumerate(cities_list):
            if type(city) is not str:
                print("line no: ", i, " city: ", city)
            if city in place_name:
                return itl_cities["country_name_de"].iloc[i]
        return pd.NA
    else:
        return pd.NA


def check_itl_countries(place_name):
    cities_list = itl_cities["city_name_de"]
    if isinstance(place_name, str):
        return any(city in place_name for city in cities_list)
    return False


def check_birthplace(place_name):
    place_name = place_name.str.strip()
    not_in_countries_mask = ~place_name.apply(check_ger_countries)
    not_in_cities_mask = ~place_name.apply(check_ger_cities)
    not_in_itl_mask = ~place_name.apply(check_itl_countries)
    matches = not_in_countries_mask & not_in_cities_mask & not_in_itl_mask
    n_issues = sum(matches)
    return matches, n_issues


def crosscheck_grade(grade_latin, grade_name):
    mapping = {
        "Summa cum laude": "Mit Auszeichnung",
        "Magma cum laude": "Sehr gut",
        "Cum laude": "Gut",
    }
    matches = pd.Series([False] * len(grade_latin), dtype=bool)
    for i, (latin, name) in enumerate(zip(grade_latin, grade_name)):
        naflag_name = False
        naflag_latin = False
        if not pd.isna(name):
            name = name.strip()
        else:
            naflag_name = True
        if not pd.isna(latin):
            latin = latin.strip()
        else:
            naflag_latin = True
        if naflag_name and naflag_latin:
            continue
        if latin in mapping:
            matches.iloc[i] = mapping[latin] != name
        else:
            matches.iloc[i] = True  # Latin grade not in mapping, hence issue
    return matches, sum(matches)
