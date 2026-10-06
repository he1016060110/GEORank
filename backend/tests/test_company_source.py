import unittest

from app.services.company_source import (
    EVIDENCE_MARKER,
    build_company_evidence,
    build_company_source,
    extract_source_document,
    prepare_company_documents,
)


class CompanySourceTests(unittest.TestCase):
    def test_css_over_5000_chars_does_not_hide_company_body(self):
        html = '<html><head><style>' + '.hidden{color:red;}' * 600 + '</style></head><body><main><h1>Example Sensors</h1><p>Example Sensors designs infrared detectors and MEMS components for industrial temperature monitoring and medical instruments.</p></main></body></html>'
        text = build_company_evidence(html, source_url='https://example.test/')
        self.assertIn('designs infrared detectors', text)
        self.assertNotIn('.hidden', text)
        self.assertTrue(text.startswith(EVIDENCE_MARKER + '\n'))

    def test_navigation_footer_cookie_and_language_widgets_removed(self):
        html = '''<html><body>
        <header>HEADER-NOISE</header><nav>NAV-NOISE</nav>
        <div class="language-switcher">ENGLISH-MENU 中文 日本語</div>
        <div id="cookie-consent">COOKIE-NOISE Accept</div>
        <main><h1>Detector Products</h1><p>Infrared thermopile modules measure non-contact temperatures using integrated CMOS-MEMS detector technology.</p><button>BUY-NOW</button></main>
        <footer>FOOTER-NOISE</footer><form>INQUIRY-NOISE</form>
        </body></html>'''
        source = extract_source_document(html)
        for noise in ['HEADER-NOISE', 'NAV-NOISE', 'ENGLISH-MENU', 'COOKIE-NOISE', 'FOOTER-NOISE', 'INQUIRY-NOISE', 'BUY-NOW']:
            self.assertNotIn(noise, source['text'])
        self.assertIn('CMOS-MEMS', source['text'])
        self.assertIn('Detector Products', source['text'])

    def test_main_preferred_but_plain_body_fallback_preserved(self):
        html = '<body><aside>ASIDE-NOISE</aside><main><p>MAIN-EVIDENCE company content and infrared technology are here.</p></main></body>'
        self.assertNotIn('ASIDE-NOISE', extract_source_document(html)['text'])
        self.assertIn('MAIN-EVIDENCE', extract_source_document(html)['text'])
        self.assertIn('BODY-EVIDENCE', extract_source_document('<body><p>BODY-EVIDENCE without any semantic main wrapper.</p></body>')['text'])

    def test_nested_main_and_repeated_slides_not_duplicated(self):
        html = '<main><div role="main"><p>Repeated descriptive product paragraph for an infrared sensing business.</p><p>Repeated descriptive product paragraph for an infrared sensing business.</p></div></main>'
        source = extract_source_document(html)
        self.assertEqual(source['text'].count('Repeated descriptive product paragraph'), 1)

    def test_json_ld_metadata_survives_script_removal(self):
        html = '''<head><title>Example Site</title><meta name="description" content="Infrared detector developer"><script>JAVASCRIPT-NOISE()</script><script type="application/ld+json">{"@context":"https://schema.org","@graph":[{"@type":"Organization","legalName":"Example Sensor Ltd","name":"EXAMPLE","description":"Develops detector modules.","address":{"@type":"PostalAddress","addressLocality":"Shanghai"}},{"@type":"Product","name":"IR-ONE","category":"infrared thermopile","description":"A detector module."}]}</script></head><body><main><p>Detectors and signal conditioning modules for industrial applications.</p></main></body>'''
        source = extract_source_document(html, 'https://example.test/about', role='about')
        self.assertEqual(source['meta']['description'], 'Infrared detector developer')
        self.assertEqual(source['organizations'][0]['legalName'], 'Example Sensor Ltd')
        self.assertEqual(source['organizations'][0]['address']['addressLocality'], 'Shanghai')
        evidence = build_company_evidence([source])
        self.assertIn('IR-ONE', evidence)
        self.assertIn('Example Sensor Ltd', evidence)
        self.assertNotIn('JAVASCRIPT-NOISE', evidence)
        self.assertNotIn('application/ld+json', source['text'])

    def test_source_urls_and_each_later_page_fit_bounded_budget(self):
        pages = [
            {'url': 'https://example.test/', 'role': 'homepage', 'html': '<main>' + '<p>Homepage generic introduction. </p>' * 1000 + '</main>'},
            {'url': 'https://example.test/company-profile/', 'role': 'about', 'html': '<main><p>ABOUT-IDENTITY Example Sensor Ltd develops thermopile detector chips and NDIR gas sensing modules.</p></main>'},
            {'url': 'https://example.test/ir-one/', 'role': 'product', 'html': '<main><p>PRODUCT-SKU IR-ONE has a digital temperature compensation interface for equipment manufacturers.</p></main>'},
        ]
        evidence = build_company_evidence(pages, max_chars=12000)
        self.assertLessEqual(len(evidence), 12000)
        self.assertIn('ABOUT-IDENTITY', evidence)
        self.assertIn('PRODUCT-SKU', evidence)
        for page in pages:
            self.assertIn(page['url'], evidence)
        self.assertLess(evidence.index('/company-profile/'), evidence.index('URL=https://example.test/;'))

    def test_company_intro_name_candidates_remain_source_attributed(self):
        source = extract_source_document('<main><p>示例芯片有限公司是一家专注于红外传感器与集成电路研发的公司，为医疗设备和工业仪器提供模块。</p></main>', 'https://example.test/about')
        self.assertEqual(source['name_candidates'][0]['value'], '示例芯片有限公司')
        self.assertEqual(source['name_candidates'][0]['url'], 'https://example.test/about')
        self.assertIn('红外传感器', source['name_candidates'][0]['quote'])

    def test_no_false_company_name_from_product_page_heading(self):
        source = extract_source_document('<main><h1>IR-ONE Detector Module</h1><p>IR-ONE is an infrared detector module with a serial digital interface for device manufacturers.</p></main>')
        self.assertEqual(source['name_candidates'], [])

    def test_invalid_json_ld_is_warning_not_lost_text(self):
        source = extract_source_document('<script type="application/ld+json">{bad-json}</script><main>Useful company facts about CMOS and MEMS technology.</main>')
        self.assertIn('json_ld_invalid', source['warnings'])
        self.assertIn('CMOS and MEMS', source['text'])

    def test_empty_and_navigation_only_have_no_main_evidence(self):
        source = extract_source_document('<nav>Home Product 中文 English</nav><footer>Copyright</footer>')
        self.assertEqual(source['text'], '')
        self.assertIn('no_main_text', source['warnings'])
        self.assertEqual(prepare_company_documents([]), [])
        self.assertIn('没有可用正文证据', build_company_evidence([]))

    def test_evidence_is_idempotent_for_ai_client_boundary(self):
        evidence = build_company_evidence('<main><p>Example company designs temperature monitoring sensor components.</p></main>')
        self.assertEqual(build_company_evidence(evidence), evidence)

    def test_source_boundary_exposes_text_sources_and_replayable_hashes(self):
        source = build_company_source('<main><p>Company infrared MEMS detector designs for medical applications.</p></main>', source_url='https://example.test/about')
        self.assertEqual(source['source_urls'], ['https://example.test/about'])
        self.assertIn('Company infrared', source['body_text'])
        self.assertTrue(source['text'].startswith(EVIDENCE_MARKER))
        self.assertEqual(len(source['text_sha256']), 64)
        self.assertEqual(len(source['evidence_sha256']), 64)
        again = build_company_source(source['documents'])
        self.assertEqual(source['text_sha256'], again['text_sha256'])
        self.assertEqual(source['evidence_sha256'], again['evidence_sha256'])

    def test_small_budget_rejected_instead_of_invalid_partial_sources(self):
        with self.assertRaises(ValueError):
            build_company_evidence('<main>Company.</main>', max_chars=200)


if __name__ == '__main__':
    unittest.main()
