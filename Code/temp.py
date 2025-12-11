def robust_primefaces_download(page_url, component_id, outfile="download.bin", session=None, save_ajax_dump="ajax_response.html", 
                               download_dir="/Users/jedrek/Documents/Studium Volkswirschaftslehre/3. Semester/Long-run dynamics of wealth inequalities/Paper/Data/CBOS numerical"):
    sess = session or requests.Session()
    sess.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })

    # 1) GET page
    r = sess.get(page_url)
    r.raise_for_status()
    doc = LH.fromstring(r.text)

    # 2) find the relevant <form> (tries to match the form that contains the component id)
    form = None
    forms = doc.xpath("//form")
    if not forms:
        # save the fetched page for debugging and raise a clear error
        try:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
            tf.write(r.content)
            tf.flush()
            tf.close()
            page_dump = tf.name
        except Exception:
            page_dump = save_ajax_dump
        raise RuntimeError(f"No <form> elements found on page {page_url}. Saved page to {page_dump}")
    for f in forms:
        # if component_id appears in this form's markup, choose it
        if component_id.split(':')[0] in (f.get("id") or "") or component_id in etree.tostring(f, encoding="unicode"):
            form = f
            break
    if form is None and forms:
        form = forms[0]

    # compute action
    action = form.get("action") or page_url
    action = urljoin(page_url, action)

    # Keep a local outfile variable so it's always defined in this scope
    _outfile = outfile

    # 3) collect all form inputs (hidden or visible) to include in payload
    payload = {}
    for inp in form.xpath(".//input"):
        name = inp.get("name")
        if not name:
            continue
        # Prefer existing value, otherwise empty string
        value = inp.get("value") or ""
        payload[name] = value

    # Ensure ViewState present
    vs_name = "javax.faces.ViewState"
    if vs_name not in payload:
        v = doc.xpath("//input[@name='javax.faces.ViewState']/@value")
        if v:
            payload[vs_name] = v[0]

    # 4) add PrimeFaces/JSF ajax parameters (typical)
    payload.update({
        "javax.faces.partial.ajax": "true",
        "javax.faces.source": component_id,
        "javax.faces.partial.execute": component_id,
        # the onclick you pasted had u:@widgetVar(downloadPackagePopup) -> render that widget
        "javax.faces.partial.render": "@widgetVar(downloadPackagePopup)",
    })
    # include the component param itself (many implementations accept empty string)
    # try both: empty string (most common) and the id string — server often accepts one of them
    payload.setdefault(component_id, "")

    headers_ajax = {
        "Faces-Request": "partial/ajax",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Referer": page_url,
    }

    # 5) POST the AJAX request
    resp = sess.post(action, data=payload, headers=headers_ajax, allow_redirects=True)
    # save the raw response for debugging; try the provided path, else fall back to a temp file
    save_path = save_ajax_dump
    try:
        with open(save_path, "wb") as fh:
            fh.write(resp.content)
    except Exception:
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        try:
            tf.write(resp.content)
            tf.flush()
        finally:
            tf.close()
        save_path = tf.name

    # quick checks: status, redirect to login, presence of login marker
    if resp.status_code >= 400:
        raise RuntimeError(f"AJAX POST returned status {resp.status_code}. Saved response to {save_path}")
    txt = resp.text

    # detect likely redirect-to-login or missing-auth: look for common words
    login_markers = ["login", "sign in", "authenticate", "csrf", "session expired"]
    low = txt.lower()
    if any(marker in low for marker in login_markers):
        # still save and raise informative error
        raise RuntimeError(f"AJAX response looks like a login/expired session page. Saved response to {save_path}")

    # 6) try many extraction strategies (XML partial response, script tags, JSON-like, plain URL)
    download_url = None

    # A) If server returned JSF partial-response XML, parse it and look for extension/update/script nodes
    try:
        root = etree.fromstring(resp.content)
        # look for <extension> nodes whose text may include apiDownloadLink or full URL
        for ext in root.findall(".//extension"):
            if ext.text:
                s = ext.text.strip()
                # unescape html entities
                s2 = html.unescape(s)
                m = re.search(r'(https?://[^"\'>\s\\]+)', s2)
                if m:
                    download_url = m.group(1)
                    break
                # try js function
                m2 = re.search(r'apiDownloadLink["\']?\s*[:=]\s*["\']([^"\']+)["\']', s2)
                if m2:
                    download_url = m2.group(1); break
        # also check <update> sections (CDATA often inside)
        if download_url is None:
            for upd in root.findall(".//update"):
                text = "".join(upd.itertext()).strip()
                if text:
                    s2 = html.unescape(text)
                    m = re.search(r'(https?://[^"\'>\s\\]+)', s2)
                    if m:
                        download_url = m.group(1)
                        break
    except Exception:
        # not XML or parse failed — continue to other strategies
        pass

    # B) search for refreshAndClickDownloadLink("https://...")
    if not download_url:
        m = re.search(r'refreshAndClickDownloadLink\(\s*["\']([^"\']+)["\']\s*\)', txt)
        if m:
            download_url = html.unescape(m.group(1))

    # C) search for apiDownloadLink in JSON-like or JS
    if not download_url:
        m = re.search(r'apiDownloadLink["\']?\s*[:=]\s*["\']([^"\']+)["\']', txt)
        if m:
            download_url = html.unescape(m.group(1))

    # D) fallback: look for any https link with expected extensions or /download path
    if not download_url:
        m = re.search(r'(https?://[^"\'>\s\\]+(?:/download|\.zip|\.dta|\.csv|\.xls|\.xlsx)[^"\'>\s\\]*)', txt)
        if m:
            download_url = html.unescape(m.group(1))

    if not download_url:
        # Save snippet and raise for user debugging
        snippet = txt[:2000].replace("\n", " ")
        raise RuntimeError(f"Couldn't find a download URL in AJAX response. Saved response to {save_path}. First 2000 chars:\n{snippet}")

    # normalize relative URLs
    download_url = urljoin(page_url, download_url)
    print("Found download URL:", download_url)

    # 7) download the actual file (stream)
    # Ensure download directory exists and is writable
    try:
        os.makedirs(download_dir, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Could not create download directory {download_dir}: {e}")
    if not os.access(download_dir, os.W_OK):
        raise RuntimeError(f"Download directory is not writable: {download_dir}")

    with sess.get(download_url, stream=True) as r2:
        r2.raise_for_status()
        # examine content-type to detect HTML responses (likely errors/login pages)
        content_type = r2.headers.get("content-type", "").lower()
        if content_type.startswith("text/html"):
            # save the HTML to a temp file for debugging and raise
            try:
                tf_err = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
                tf_err.write(r2.content)
                tf_err.flush()
                tf_err.close()
                raise RuntimeError(f"Download URL returned HTML (not a file). Saved HTML to {tf_err.name}")
            except Exception:
                raise RuntimeError("Download URL returned HTML (not a file) and saving the HTML failed.")

        # try to pick filename from headers
        cd = r2.headers.get("content-disposition", "")
        candidate = None
        if "filename=" in cd:
            import re as _re
            fn = _re.search(r'filename\*?=(?:UTF-8\'\')?["\']?([^"\';]+)', cd)
            if fn:
                candidate = fn.group(1)

        # if no filename from header, parse from URL
        if not candidate:
            p = urlparse(download_url)
            candidate = os.path.basename(unquote(p.path))

        # fallback: try to find a .dta or .pdf in the URL
        if not candidate or not os.path.splitext(candidate)[1]:
            m = re.search(r'([^/\\]+\.(?:dta|pdf))(?:$|[?;])', download_url, flags=re.IGNORECASE)
            if m:
                candidate = m.group(1)

        # sanitize and ensure extension is .dta or .pdf
        candidate = candidate or outfile
        candidate = os.path.basename(candidate)
        name, ext = os.path.splitext(candidate)
        ext = ext.lower()
        allowed = {".dta", ".pdf"}
        if ext not in allowed:
            # if we can detect .dta or .pdf in the URL, use that; else default to .dta
            m2 = re.search(r'\.(dta|pdf)(?:$|[?;])', download_url, flags=re.IGNORECASE)
            if m2:
                ext = "." + m2.group(1).lower()
            else:
                ext = ".dta"
            candidate = name + ext

        # final target path (avoid name collisions)
        target_path = os.path.join(download_dir, candidate)
        base, extension = os.path.splitext(target_path)
        i = 1
        while os.path.exists(target_path):
            target_path = f"{base}_{i}{extension}"
            i += 1

        # write to temporary file in target dir then atomically move
        try:
            tf = tempfile.NamedTemporaryFile(delete=False, dir=download_dir, suffix=".part")
            try:
                for chunk in r2.iter_content(8192):
                    if chunk:
                        tf.write(chunk)
                tf.flush()
            finally:
                tf.close()
            os.replace(tf.name, target_path)
            _outfile = target_path
        except Exception as e:
            # fallback: write to system temp
            try:
                tf2 = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                try:
                    for chunk in r2.iter_content(8192):
                        if chunk:
                            tf2.write(chunk)
                    tf2.flush()
                finally:
                    tf2.close()
                _outfile = tf2.name
            except Exception:
                raise RuntimeError(f"Failed to save downloaded file to {download_dir} and system temp. Error: {e}")

    return _outfile










comp = "datasetForm:tabView:filesTable:1:j_idt644:tabularOriginalDownloadPopupButton"
    try:
        res = robust_primefaces_download(page, comp, outfile="data.dta")
        print("Downloaded to", res)
    except Exception as e:
        print("ERROR:", e)
        # ajax response saved as ajax_response.html for inspection