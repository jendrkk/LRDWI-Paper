"""
LIS LISSY Web Interface Automizer
=================================
Automates job submission on the LIS Data Center LISSY web application
(https://webui.lisdatacenter.org) using Selenium with Safari WebDriver.

The ZK framework generates dynamic element IDs per session, so this module
uses stable CSS selectors and XPaths based on element classes, titles, and
structural position.

Credentials:
    - User ID: ``jslowi`` (hardcoded)
    - Password: read automatically from the RTF file at
      ``<project_root>/../../password.rtf``

Usage:
    from tools.lis_automizer import LISAutomizer

    with LISAutomizer() as lis:
        result = lis.submit_job(
            code='print("hello")',
            job_title="my test job",
            project="LIS",
            package="R",
        )
        print(result)
"""

import os
import re
import subprocess
import time
import logging
from pathlib import Path
from typing import Optional

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
)

logger = logging.getLogger(__name__)

# ── Credentials ──────────────────────────────────────────────────────────────
LIS_USER_ID = "" # set your user ID here

# Password file path (relative to this script's location):
# tools/ → Code/ → LRDWI-Paper/ → local_repo/ → Paper/ → Long-run.../password.rtf
_THIS_DIR = Path(__file__).resolve().parent
_PASSWORD_FILE = _THIS_DIR.parent.parent.parent.parent.parent / "password.rtf"


def _read_password_from_rtf(rtf_path: Path = _PASSWORD_FILE) -> str:
    """Extract plain text from an RTF file using macOS textutil."""
    rtf_path = Path(rtf_path)
    if not rtf_path.exists():
        raise FileNotFoundError(
            f"Password file not found: {rtf_path}\n"
            "Expected at: <project_data_root>/password.rtf"
        )
    result = subprocess.run(
        ["textutil", "-convert", "txt", "-stdout", str(rtf_path)],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()

# ── Stable selectors (ZK IDs are dynamic, so we rely on classes / titles) ────
LISSY_URL = "https://webui.lisdatacenter.org/userinterface/"

# The main code editor textarea (has z-flex-item class, unique on the page)
SEL_CODE_TEXTAREA = "textarea.z-textbox.z-flex-item"

# The "submit job" button carries a stable title attribute
SEL_RUN_BUTTON = 'a.z-toolbarbutton[title="submit job to postoffice for execution"]'

# Subject / job-title input: find the "Subject" label then its sibling cell's <input>
XPATH_SUBJECT_INPUT = (
    "//span[@class='z-label' and text()='Subject']"
    "/ancestor::td/following-sibling::td//input[@class='z-textbox']"
)

# Project <select> is the first z-select on the page; Package is the second
SEL_PROJECT_SELECT = "select.z-select"

# Status label that tells us the job is ready or has been submitted
XPATH_STATUS_LABEL = (
    "//span[contains(@class,'z-label') and ("
    "contains(text(),'Job can be sent') or "
    "contains(text(),'submitted') or "
    "contains(text(),'Job was sent'))]"
)

# "recent jobs" tab
XPATH_RECENT_JOBS_TAB = "//span[contains(@class,'z-tab-text') and contains(.,'recent jobs')]"

# Job list rows in the "recent jobs" panel
SEL_JOB_LIST_ROW = "tr.z-listitem"

# ── Login page selectors (ZK-based login form inside div.windowlogin) ────────
# Username: the z-textbox with type="text" inside the windowlogin panel
SEL_LOGIN_USER = 'div.windowlogin input.z-textbox[type="text"]'
# Password: the z-textbox with type="password" inside the windowlogin panel
SEL_LOGIN_PASS = 'div.windowlogin input.z-textbox[type="password"]'
# Connect button: z-button with text "connect"
XPATH_LOGIN_CONNECT = '//button[contains(@class,"z-button") and contains(text(),"connect")]'


class LISAutomizer:
    """Drives the LIS LISSY web UI via Selenium (Safari WebDriver)."""

    def __init__(
        self,
        implicit_wait: int = 10,
        page_load_timeout: int = 60,
    ):
        """
        Parameters
        ----------
        implicit_wait : int
            Selenium implicit wait in seconds.
        page_load_timeout : int
            Maximum seconds to wait for page loads.
        """
        self.user_id = LIS_USER_ID
        self.password = _read_password_from_rtf()
        self.implicit_wait = implicit_wait
        self.page_load_timeout = page_load_timeout
        self.driver: Optional[webdriver.Safari] = None

    # ── Context manager ──────────────────────────────────────────────────
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.quit()

    # ── Lifecycle ────────────────────────────────────────────────────────
    def start(self):
        """Launch Safari and navigate to LISSY, then log in."""
        self.driver = webdriver.Safari()
        self.driver.implicitly_wait(self.implicit_wait)
        self.driver.set_page_load_timeout(self.page_load_timeout)

        self.driver.get(LISSY_URL)
        logger.info("Navigated to %s", LISSY_URL)

        # Always log in (fresh session = login page)
        self._handle_login()

    def quit(self):
        """Close the browser."""
        if self.driver:
            self.driver.quit()
            self.driver = None

    # ── Login handling ───────────────────────────────────────────────────
    def _handle_login(self):
        """Log into the LISSY ZK web interface and wait for the editor.

        The login page (div.windowlogin) has:
          - input.z-textbox[type='text']     → user ID
          - input.z-textbox[type='password'] → password
          - button.z-button "connect"        → submit
        After clicking connect, the page reloads into the editor.
        """
        wait = WebDriverWait(self.driver, 20)

        # Check if we're already past the login page
        try:
            wait_quick = WebDriverWait(self.driver, 8)
            wait_quick.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, SEL_CODE_TEXTAREA))
            )
            logger.info("Already authenticated – LISSY editor is visible.")
            return
        except TimeoutException:
            pass

        logger.info("Login page detected – authenticating as '%s'…", self.user_id)

        # Wait for the login form to render (ZK is JS-heavy)
        user_input = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, SEL_LOGIN_USER))
        )
        pass_input = self.driver.find_element(By.CSS_SELECTOR, SEL_LOGIN_PASS)

        user_input.clear()
        user_input.send_keys(self.user_id)
        pass_input.clear()
        pass_input.send_keys(self.password)

        # Click the "connect" button.  The click may trigger a full page
        # navigation; Safari WebDriver can throw during/after that, so we
        # wrap the click itself in a broad try/except as well.
        connect_btn = self.driver.find_element(By.XPATH, XPATH_LOGIN_CONNECT)
        try:
            connect_btn.click()
        except Exception as e:
            logger.info("Click raised %s (expected during page navigation)", e)
        logger.info("Clicked 'connect' – waiting for editor to load…")

        # Give the page time to start navigating / reloading.
        # During navigation Safari WebDriver may throw errors on any element
        # query, so we must poll manually with broad exception handling.
        time.sleep(3)

        max_wait_login = 60
        poll_interval = 2
        elapsed = 0
        while elapsed < max_wait_login:
            try:
                el = self.driver.find_element(By.CSS_SELECTOR, SEL_CODE_TEXTAREA)
                if el:
                    logger.info("Login successful – LISSY editor loaded.")
                    return
            except Exception:
                # During page transition any exception is expected – keep polling
                pass
            time.sleep(poll_interval)
            elapsed += poll_interval

        # If we get here, the editor never appeared
        error_msg = "(could not inspect page)"
        try:
            warn = self.driver.find_element(By.CSS_SELECTOR, "span.warn.z-label")
            error_msg = warn.text
        except Exception:
            pass
        raise RuntimeError(
            f"Login failed – the LISSY editor did not appear after "
            f"{max_wait_login}s. Login message: {error_msg}  |  "
            f"URL: {self.driver.current_url}"
        )

    # ── Core actions ─────────────────────────────────────────────────────
    def _select_dropdown(self, index: int, value: str, label: str):
        """Select a value from the Nth z-select dropdown with retry for staleness."""
        for attempt in range(3):
            try:
                time.sleep(0.5)
                selects = self.driver.find_elements(By.CSS_SELECTOR, SEL_PROJECT_SELECT)
                if len(selects) <= index:
                    raise RuntimeError(f"{label} dropdown not found.")
                Select(selects[index]).select_by_visible_text(value)
                logger.info("%s set to %s", label, value)
                return
            except StaleElementReferenceException:
                logger.debug("Stale element on %s attempt %d, retrying…", label, attempt)
                time.sleep(1)
        raise RuntimeError(f"Could not set {label} to {value} after 3 attempts (stale elements).")

    def set_project(self, project: str = "LIS"):
        """Select the project from the dropdown (LIS, LWS, LWSPRE, etc.)."""
        self._select_dropdown(0, project, "Project")

    def set_package(self, package: str = "R"):
        """Select the statistical package (R, Stata, SAS, SPSS)."""
        self._select_dropdown(1, package, "Package")

    def set_subject(self, title: str):
        """Set the job subject / title."""
        field = self.driver.find_element(By.XPATH, XPATH_SUBJECT_INPUT)
        field.clear()
        field.send_keys(title)
        # Trigger ZK's onChange by tabbing out
        field.send_keys(Keys.TAB)
        logger.info("Subject set to '%s'", title)

    def set_code(self, code: str):
        """Paste code into the main editor textarea.

        Uses JavaScript to set the value reliably (ZK textarea may ignore
        normal send_keys for large text), then triggers ZK's change event.
        """
        textarea = WebDriverWait(self.driver, 15).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, SEL_CODE_TEXTAREA))
        )
        # Clear + set via JS to handle large code blocks efficiently
        self.driver.execute_script(
            """
            var ta = arguments[0];
            ta.focus();
            ta.value = arguments[1];
            // Fire change/input events so the ZK framework picks it up
            ta.dispatchEvent(new Event('input',  {bubbles: true}));
            ta.dispatchEvent(new Event('change', {bubbles: true}));
            ta.dispatchEvent(new Event('blur',   {bubbles: true}));
            """,
            textarea,
            code,
        )
        time.sleep(0.5)  # let ZK process the event
        logger.info("Code pasted (%d chars)", len(code))

    def click_run(self):
        """Click the submit/run button."""
        btn = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, SEL_RUN_BUTTON))
        )
        btn.click()
        logger.info("Run button clicked – job submitted.")
        time.sleep(1)  # brief pause for ZK processing

    def submit_job(
        self,
        code: str,
        job_title: str = "automated job",
        project: str = "LIS",
        package: str = "R",
        wait_for_result: bool = True,
        poll_interval: int = 10,
        max_wait: int = 300,
    ) -> Optional[str]:
        """Full workflow: set fields, paste code, submit, optionally wait for output.

        Parameters
        ----------
        code : str
            The R/Stata/SAS/SPSS code to submit.
        job_title : str
            Job subject line.
        project : str
            LIS project name.
        package : str
            Statistical package.
        wait_for_result : bool
            If True, poll the "recent jobs" tab until the job finishes.
        poll_interval : int
            Seconds between polls when waiting for the result.
        max_wait : int
            Maximum seconds to wait for the result.

        Returns
        -------
        str or None
            Job result text (if wait_for_result is True), else None.
        """
        self.set_project(project)
        self.set_package(package)
        self.set_subject(job_title)
        self.set_code(code)
        self.click_run()

        # Handle possible confirmation dialog from ZK
        self._dismiss_confirmation_dialog()

        if wait_for_result:
            return self._wait_for_result(
                job_title=job_title,
                poll_interval=poll_interval,
                max_wait=max_wait,
            )
        return None

    # ── Result retrieval ─────────────────────────────────────────────────
    def _wait_for_result(
        self,
        job_title: str,
        poll_interval: int = 10,
        max_wait: int = 300,
    ) -> Optional[str]:
        """Switch to 'recent jobs' tab and poll until the latest job is done."""
        # Click the "recent jobs" tab
        try:
            tab = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, XPATH_RECENT_JOBS_TAB))
            )
            tab.click()
            time.sleep(2)
        except TimeoutException:
            logger.warning("Could not find 'recent jobs' tab.")
            return None

        elapsed = 0
        while elapsed < max_wait:
            # Look for the most recent job row
            rows = self.driver.find_elements(By.CSS_SELECTOR, SEL_JOB_LIST_ROW)
            if rows:
                first_row = rows[0]
                cells = first_row.find_elements(By.CSS_SELECTOR, "td div")
                # The first cell has a status icon; check if it's "job_done"
                status_imgs = first_row.find_elements(By.TAG_NAME, "img")
                if status_imgs:
                    src = status_imgs[0].get_attribute("src") or ""
                    if "job_done" in src:
                        logger.info("Job completed!")
                        # Click on the row via JS (Safari may refuse native click)
                        self.driver.execute_script("arguments[0].click();", first_row)
                        time.sleep(2)
                        return self._extract_result_text()

            time.sleep(poll_interval)
            elapsed += poll_interval
            # Refresh the job list by clicking the recent-jobs tab again
            try:
                tab = self.driver.find_element(By.XPATH, XPATH_RECENT_JOBS_TAB)
                tab.click()
                time.sleep(1)
            except NoSuchElementException:
                pass

        logger.warning("Timed out waiting for job result after %ds", max_wait)
        return None

    def _extract_result_text(self) -> str:
        """Try to grab the result text from the opened result panel."""
        # Results often appear in a new tab panel with a textarea or pre/div
        try:
            # Look for result textarea / code area in the right panel
            result_els = self.driver.find_elements(
                By.CSS_SELECTOR, "div.z-tabpanel textarea, div.z-tabpanel pre"
            )
            for el in result_els:
                txt = el.get_attribute("value") or el.text
                if txt and txt.strip():
                    return txt.strip()

            # Fallback: look for any visible z-tabpanel content
            panels = self.driver.find_elements(By.CSS_SELECTOR, "div.z-tabpanel")
            for panel in panels:
                if panel.is_displayed() and panel.text.strip():
                    return panel.text.strip()
        except Exception as e:
            logger.warning("Error extracting result: %s", e)

        return ""

    def _dismiss_confirmation_dialog(self):
        """Dismiss any ZK confirmation/messagebox that pops up after clicking run."""
        time.sleep(1)
        try:
            ok_btns = self.driver.find_elements(
                By.CSS_SELECTOR, "div.z-messagebox-window button.z-button"
            )
            for btn in ok_btns:
                if btn.is_displayed():
                    btn.click()
                    time.sleep(0.5)
                    return
        except Exception:
            pass

    # ── Helpers ──────────────────────────────────────────────────────────
    def _find_first(self, *locators):
        """Return the first element matched by any of the given locators, or None."""
        for by, value in locators:
            try:
                el = self.driver.find_element(by, value)
                if el:
                    return el
            except NoSuchElementException:
                continue
        return None

    def get_status_text(self) -> str:
        """Read the current status label (e.g. 'Job can be sent to the server.')."""
        try:
            el = self.driver.find_element(By.XPATH, XPATH_STATUS_LABEL)
            return el.text.strip()
        except NoSuchElementException:
            return ""


# ── Convenience function for quick one-shot jobs ─────────────────────────────
def run_lis_job(
    code: str,
    job_title: str = "automated job",
    project: str = "LIS",
    package: str = "R",
    wait_for_result: bool = True,
    max_wait: int = 300,
) -> Optional[str]:
    """Submit a single job and optionally wait for the result.

    Credentials are read automatically (user: jslowi, password from RTF file).

    Parameters
    ----------
    code : str
        Code to execute on the LIS server.
    job_title : str
        Subject / title for the job.
    project : str
        LIS project (LIS, LWS, etc.).
    package : str
        Statistical package (R, Stata, SAS, SPSS).
    wait_for_result : bool
        Wait for the job to finish and return the output.
    max_wait : int
        Max seconds to wait for the result.

    Returns
    -------
    str or None
        Job output text, or None if not waiting.
    """
    with LISAutomizer() as lis:
        return lis.submit_job(
            code=code,
            job_title=job_title,
            project=project,
            package=package,
            wait_for_result=wait_for_result,
            max_wait=max_wait,
        )

def read_job_from_txt(file_path: str) -> str:
    """Read code from a .txt file to submit as a job."""
    with open(file_path, 'r') as f:
        return f.read()

def main():
    # Example usage: run a simple R job and print the result
    code = """
    print("hello from job 2")
    """
    result = run_lis_job(
        code=code,
        job_title="Test job from LISAutomizer",
        project="LIS",
        package="R",
        wait_for_result=False,
        max_wait=300,
    )
    print("Job result:")
    print(result)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    main()