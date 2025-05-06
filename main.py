#@title Automated Redirect Mapping Tool
"""
Automated Redirect Mapper using Google Gemini API - Optimized for Google Colab
This script helps create redirect mappings for SEO purposes using AI assistance.
"""

import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from google import genai
import requests
from urllib.parse import urlparse
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import time
import hashlib
import pickle
import os
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class CacheEntry:
    """Cache entry for API responses."""
    response: Dict
    timestamp: datetime

    def is_expired(self, max_age_hours: int = 24) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() - self.timestamp > timedelta(hours=max_age_hours)


class RedirectMapper:
    def __init__(self, api_key: str, use_cache: bool = True, batch_size: int = 25):
        """
        Initialize the RedirectMapper with Gemini API key.

        Args:
            api_key: Google Gemini API key
            use_cache: Whether to use caching for API responses
            batch_size: Number of URLs to process in each API call
        """
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.0-flash"  # Most cost-efficient model
        self.use_cache = use_cache
        self.batch_size = batch_size
        self.cache_file = "redirect_mapper_cache.pkl"
        self.cache = self._load_cache() if use_cache else {}
        self.url_structure_threshold = 0.7  # Threshold for URL structure matching
        self.ai_confidence_threshold = 0.7  # Threshold for AI matching before fallback

    def _load_cache(self) -> Dict:
        """Load cache from file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
        return {}

    def _save_cache(self):
        """Save cache to file."""
        if self.use_cache:
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.cache, f)
            except Exception as e:
                print(f"Error saving cache: {e}")

    def _get_cache_key(self, old_urls: List[str], new_urls: List[str]) -> str:
        """Generate cache key for URL lists."""
        combined = str(sorted(old_urls)) + str(sorted(new_urls))
        return hashlib.md5(combined.encode()).hexdigest()

    def extract_urls_from_file(self, file_path: str) -> List[str]:
        """Extract URLs from various file formats with progress bar."""
        file_path = Path(file_path)
        urls = []

        print(f"📂 Reading URLs from {file_path.name}")

        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
            # Look for URL columns
            url_columns = [col for col in df.columns if 'url' in col.lower() or 'path' in col.lower()]
            if url_columns:
                for col in url_columns:
                    urls.extend(df[col].dropna().tolist())
            else:
                # If no URL columns found, take the first column
                urls = df.iloc[:, 0].dropna().tolist()

        elif file_path.suffix.lower() == '.txt':
            with open(file_path, 'r') as f:
                urls = [line.strip() for line in f.readlines() if line.strip()]

        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    urls = data
                elif isinstance(data, dict):
                    # Extract URLs from dict values
                    urls = self._extract_urls_from_json(data)

        print(f"✅ Extracted {len(urls)} URLs")
        return urls

    def _extract_urls_from_json(self, data: Dict, urls: List[str] = None) -> List[str]:
        """Recursively extract URLs from JSON data."""
        if urls is None:
            urls = []

        for key, value in data.items():
            if isinstance(value, str) and ('http' in value or '/' in value):
                urls.append(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and ('http' in item or '/' in item):
                        urls.append(item)
                    elif isinstance(item, dict):
                        self._extract_urls_from_json(item, urls)
            elif isinstance(value, dict):
                self._extract_urls_from_json(value, urls)

        return urls

    def extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        if url.startswith(('http://', 'https://')):
            parsed = urlparse(url)
            return f"{parsed.scheme}://{parsed.netloc}"
        return ""

    def normalize_url(self, url: str) -> str:
        """Normalize URL for comparison."""
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        parsed = urlparse(url)
        path = parsed.path.rstrip('/')

        return path if path else '/'

    def combine_domain_and_path(self, domain: str, path: str) -> str:
        """Combine domain and path to create full URL."""
        if not domain:
            return path
        return domain.rstrip('/') + '/' + path.lstrip('/')

    def analyze_url_structure(self, urls: List[str]) -> Dict[str, List[str]]:
        """Analyze URL structure and group similar URLs."""
        url_patterns = defaultdict(list)

        for url in tqdm(urls, desc="Analyzing URL structure"):
            normalized = self.normalize_url(url)
            parts = normalized.split('/')

            # Group by path depth
            depth = len([p for p in parts if p])
            url_patterns[f"depth_{depth}"].append(url)

            # Group by main category (first path segment)
            if len(parts) > 1 and parts[1]:
                url_patterns[f"category_{parts[1]}"].append(url)

        return dict(url_patterns)

    def _create_optimized_prompt(self, old_urls: List[str], new_urls: List[str]) -> str:
        """Create an optimized prompt to reduce token usage."""
        # Normalize and deduplicate URLs
        old_normalized = list(set([self.normalize_url(url) for url in old_urls]))
        new_normalized = list(set([self.normalize_url(url) for url in new_urls]))

        # Create compact representation
        prompt = f"""Map old URLs to new URLs for website migration. Return JSON only.

Old URLs ({len(old_normalized)}):
{json.dumps(old_normalized[:self.batch_size])}

New URLs ({len(new_normalized)}):
{json.dumps(new_normalized[:self.batch_size])}

Return format:
{{"redirects": [{{"old": "/path", "new": "/new-path", "conf": 0.9}}]}}

Rules:
- Match similar content/structure
- Return ALL matches, even low confidence ones
- Brief format
- No explanations"""

        return prompt

    def url_structure_matching(self, old_urls: List[str], new_urls: List[str]) -> List[Dict[str, str]]:
        """Enhanced URL structure matching algorithm."""
        redirects = []

        for old_url in tqdm(old_urls, desc="URL structure matching"):
            old_normalized = self.normalize_url(old_url)
            old_parts = old_normalized.split('/')
            old_domain = self.extract_domain(old_url)

            best_match = None
            best_score = 0
            best_match_domain = ""

            for new_url in new_urls:
                new_normalized = self.normalize_url(new_url)
                new_parts = new_normalized.split('/')
                new_domain = self.extract_domain(new_url)

                # Calculate similarity score
                score = 0
                max_length = max(len(old_parts), len(new_parts))

                # Exact path matching weight
                for i, part in enumerate(old_parts):
                    if i < len(new_parts) and part == new_parts[i]:
                        score += 1.0

                # Partial keyword matching
                old_keywords = set([k for p in old_parts for k in p.lower().split('-') if k])
                new_keywords = set([k for p in new_parts for k in p.lower().split('-') if k])

                if old_keywords and new_keywords:
                    keyword_overlap = len(old_keywords.intersection(new_keywords))
                    keyword_score = keyword_overlap / max(len(old_keywords), len(new_keywords))
                    score += keyword_score * 2.0  # Weight keyword matches

                # Depth similarity bonus
                depth_diff = abs(len(old_parts) - len(new_parts))
                depth_score = 1.0 / (1.0 + depth_diff)
                score += depth_score

                # Normalize score
                normalized_score = score / (max_length + 3)  # +3 for keyword and depth scores

                if normalized_score > best_score:
                    best_score = normalized_score
                    best_match = new_normalized
                    best_match_domain = new_domain

            if best_match and best_score >= self.url_structure_threshold:
                redirects.append({
                    "old_url": self.combine_domain_and_path(old_domain, old_normalized),
                    "new_url": self.combine_domain_and_path(best_match_domain, best_match),
                    "confidence": min(best_score, 1.0),
                    "reason": "URL structure match"
                })

        return redirects

    def generate_redirect_suggestions(self, old_urls: List[str], new_urls: List[str]) -> List[Dict[str, str]]:
        """Generate redirect suggestions with hierarchy: structure match -> AI match -> homepage fallback."""
        all_redirects = []

        # Extract domains for later use
        old_domains = {}
        new_domain = ""

        # Create a mapping of normalized URLs to original URLs
        normalized_to_original = {}
        for url in old_urls:
            normalized = self.normalize_url(url)
            normalized_to_original[normalized] = url
            domain = self.extract_domain(url)
            if domain:
                old_domains[normalized] = domain

        for url in new_urls:
            domain = self.extract_domain(url)
            if domain:
                new_domain = domain
                break  # Assume all new URLs share the same domain

        # First pass: URL structure matching
        print("🔍 Attempting URL structure matching...")
        structure_redirects = self.url_structure_matching(old_urls, new_urls)

        # Track which normalized URLs have been matched
        matched_normalized_urls = set()
        for redirect in structure_redirects:
            # Extract the normalized URL from the redirect
            old_url_normalized = self.normalize_url(redirect['old_url'])
            matched_normalized_urls.add(old_url_normalized)

        all_redirects.extend(structure_redirects)
        print(f"✅ Structure matching found {len(structure_redirects)} redirects")

        # Find unmatched URLs
        unmatched_old_urls = []
        for url in old_urls:
            normalized = self.normalize_url(url)
            if normalized not in matched_normalized_urls:
                unmatched_old_urls.append(url)

        # Second pass: AI matching for unmatched URLs
        if unmatched_old_urls:
            print(f"🤖 Processing {len(unmatched_old_urls)} unmatched URLs with AI...")
            ai_redirects = self._generate_ai_redirects(unmatched_old_urls, new_urls)

            # Process AI results and apply homepage fallback
            for i, old_url in enumerate(unmatched_old_urls):
                old_normalized = self.normalize_url(old_url)
                old_domain = old_domains.get(old_normalized, "")

                # Find the corresponding AI redirect
                ai_redirect = None
                for redirect in ai_redirects:
                    if self.normalize_url(redirect['old_url']) == old_normalized:
                        ai_redirect = redirect
                        break

                if ai_redirect and ai_redirect['confidence'] >= self.ai_confidence_threshold:
                    # Good AI match - use it
                    new_normalized = self.normalize_url(ai_redirect['new_url'])
                    all_redirects.append({
                        "old_url": self.combine_domain_and_path(old_domain, old_normalized),
                        "new_url": self.combine_domain_and_path(new_domain, new_normalized),
                        "confidence": ai_redirect['confidence'],
                        "reason": "AI mapping"
                    })
                else:
                    # Low confidence or no AI match - use homepage fallback
                    all_redirects.append({
                        "old_url": self.combine_domain_and_path(old_domain, old_normalized),
                        "new_url": new_domain if new_domain else "/",
                        "confidence": 0.4,
                        "reason": "Homepage fallback (low confidence)"
                    })

        print(f"✅ Total redirects generated: {len(all_redirects)}")
        return all_redirects

    def _generate_ai_redirects(self, old_urls: List[str], new_urls: List[str]) -> List[Dict[str, str]]:
        """Generate redirects using Gemini API (separated from main flow)."""
        ai_redirects = []
        total_batches = (len(old_urls) + self.batch_size - 1) // self.batch_size

        for i in tqdm(range(0, len(old_urls), self.batch_size), desc="AI processing batches"):
            batch_old = old_urls[i:i + self.batch_size]
            batch_new = new_urls  # Keep all new URLs for better matching

            # Check cache first
            cache_key = self._get_cache_key(batch_old, batch_new)
            if self.use_cache and cache_key in self.cache:
                cache_entry = self.cache[cache_key]
                if not cache_entry.is_expired():
                    ai_redirects.extend(cache_entry.response)
                    continue

            try:
                # Create optimized prompt
                prompt = self._create_optimized_prompt(batch_old, batch_new)

                # Make API call
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt
                )

                # Parse response
                result_text = response.text
                start_idx = result_text.find('{')
                end_idx = result_text.rfind('}') + 1

                if start_idx != -1 and end_idx != -1:
                    json_str = result_text[start_idx:end_idx]
                    redirect_data = json.loads(json_str)
                    batch_redirects = []

                    # Convert abbreviated format to full format
                    for r in redirect_data.get('redirects', []):
                        # Important: Don't filter by confidence here
                        # Let the main function handle the threshold
                        batch_redirects.append({
                            "old_url": r.get('old', ''),
                            "new_url": r.get('new', ''),
                            "confidence": r.get('conf', 0.0),
                            "reason": "AI mapping"
                        })

                    ai_redirects.extend(batch_redirects)

                    # Cache the result
                    if self.use_cache:
                        self.cache[cache_key] = CacheEntry(batch_redirects, datetime.now())

                # Rate limiting
                time.sleep(0.5)  # Avoid hitting rate limits

            except Exception as e:
                print(f"⚠️ Error in batch {i//self.batch_size + 1}: {e}")
                # Return empty results for this batch, not fallbacks
                # The main function will handle fallbacks
                for old_url in batch_old:
                    ai_redirects.append({
                        "old_url": old_url,
                        "new_url": "",
                        "confidence": 0.0,
                        "reason": "AI error"
                    })

        # Save cache
        self._save_cache()
        return ai_redirects

    def export_redirects(self, redirects: List[Dict[str, str]], output_file: str, format: str = 'csv'):
        """Export redirects to file in various formats with progress bar."""
        output_path = Path(output_file)

        print(f"💾 Exporting {len(redirects)} redirects to {format} format...")

        if format.lower() == 'csv':
            df = pd.DataFrame(redirects)
            df.to_csv(output_path, index=False)

        elif format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(redirects, f, indent=2)

        elif format.lower() == 'nginx':
            with open(output_path, 'w') as f:
                for redirect in tqdm(redirects, desc="Writing Nginx config"):
                    f.write(f"rewrite ^{redirect['old_url']}$ {redirect['new_url']} permanent;\n")

        elif format.lower() == 'apache':
            with open(output_path, 'w') as f:
                for redirect in tqdm(redirects, desc="Writing Apache config"):
                    f.write(f"Redirect 301 {redirect['old_url']} {redirect['new_url']}\n")

        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"✅ Exported to {output_path}")

    def generate_analytics_report(self, redirects: List[Dict[str, str]]) -> Dict:
      """Generate analytics report for the redirect mapping."""
      report = {
          "total_redirects": len(redirects),
          "reason_distribution": {},
          "avg_confidence": 0,
          "high_confidence_count": 0,
          "medium_confidence_count": 0,
          "fallback_count": 0
      }

      if redirects:
          confidences = [r['confidence'] for r in redirects]
          report['avg_confidence'] = sum(confidences) / len(confidences)
          report['high_confidence_count'] = len([c for c in confidences if c >= 0.9])
          report['medium_confidence_count'] = len([c for c in confidences if 0.7 <= c < 0.9])
          report['fallback_count'] = len([r for r in redirects if 'fallback' in r['reason'].lower()])

          # Reason distribution
          reasons = defaultdict(int)
          for redirect in redirects:
              reasons[redirect.get('reason', 'Unknown')] += 1
          report['reason_distribution'] = dict(reasons)

      return report


# Google Colab specific functions
def setup_colab_environment():
    """Setup Google Colab environment."""
    print("🔧 Setting up Google Colab environment...")

    # Install required packages
    print("📦 Installing required packages...")
    os.system('pip install -q -U google-genai pandas tqdm')

    print("✅ Environment setup complete!")


def main():
    """Main function for running in Google Colab or command line."""
    # Check if running in Google Colab
    try:
        import google.colab
        IN_COLAB = True
        print("🚀 Running in Google Colab")
        setup_colab_environment()
    except ImportError:
        IN_COLAB = False
        print("💻 Running locally")

    if IN_COLAB:
        # Interactive mode for Colab
        from google.colab import files
        from google.colab import userdata

        print("\n" + "="*50)
        print("🔄 AUTOMATED REDIRECT MAPPER")
        print("="*50)

        # Get API key from Colab secrets or user input
        try:
            api_key = userdata.get('GEMINI_API_KEY')
            print("✅ API key loaded from Colab secrets")
        except Exception as e:
            print("⚠️ Could not load API key from Colab secrets")
            print("Please add your GEMINI_API_KEY to Colab secrets or enter it manually")
            from getpass import getpass
            api_key = getpass("Enter your Google Gemini API key: ")

        # File upload
        print("📤 Upload your old URLs file:")
        uploaded = files.upload()
        old_file = list(uploaded.keys())[0]

        print("📤 Upload your new URLs file:")
        uploaded = files.upload()
        new_file = list(uploaded.keys())[0]

        # Generate all output formats
        print("\n📝 Generating all output formats...")
        output_formats = ['csv', 'json', 'nginx', 'apache']

        # Initialize mapper
        mapper = RedirectMapper(api_key, use_cache=True, batch_size=25)

        # Extract URLs
        old_urls = mapper.extract_urls_from_file(old_file)
        new_urls = mapper.extract_urls_from_file(new_file)

        # Generate redirects
        redirects = mapper.generate_redirect_suggestions(old_urls, new_urls)

        # Export redirects in all formats
        output_files = []
        for format in output_formats:
            output_file = f"redirects.{format}"
            mapper.export_redirects(redirects, output_file, format)
            output_files.append(output_file)

        # Generate report
        report = mapper.generate_analytics_report(redirects)
        report_file = 'redirect_analytics.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        # Display summary
        print("\n" + "="*50)
        print("📊 REDIRECT MAPPING SUMMARY")
        print("="*50)
        print(f"Total redirects: {report['total_redirects']}")
        print(f"Average confidence: {report['avg_confidence']:.2f}")
        print(f"High confidence (≥0.9): {report['high_confidence_count']}")
        print(f"Medium confidence (0.7-0.9): {report['medium_confidence_count']}")
        print(f"Low confidence - Homepage fallbacks: {report['fallback_count']}")
        print("\nReason distribution:")
        for reason, count in report['reason_distribution'].items():
            print(f"  {reason}: {count} redirects")

        # Download all files
        print("\n📥 Downloading all generated files...")

        # Download all redirect files
        for file in output_files:
            files.download(file)

        # Download report
        files.download(report_file)

        # Download cache file if it exists
        if os.path.exists(mapper.cache_file):
            files.download(mapper.cache_file)
            print("📦 Cache file downloaded for future use")

        print("\n✅ Process complete! All files downloaded:")
        print("- redirects.csv")
        print("- redirects.json")
        print("- redirects.nginx")
        print("- redirects.apache")
        print("- redirect_analytics.json")
        if os.path.exists(mapper.cache_file):
            print("- redirect_mapper_cache.pkl")

    else:
        # Command line mode
        parser = argparse.ArgumentParser(description='Automated Redirect Mapper using Gemini API')
        parser.add_argument('--api-key', required=True, help='Google Gemini API key')
        parser.add_argument('--old-urls', required=True, help='File containing old URLs')
        parser.add_argument('--new-urls', required=True, help='File containing new URLs')
        parser.add_argument('--output', default='redirects.csv', help='Output file for redirects')
        parser.add_argument('--format', choices=['csv', 'json', 'nginx', 'apache'],
                            default='csv', help='Output format')
        parser.add_argument('--report', action='store_true', help='Generate analytics report')
        parser.add_argument('--batch-size', type=int, default=25, help='Batch size for API calls')
        parser.add_argument('--no-cache', action='store_true', help='Disable caching')

        args = parser.parse_args()

        # Initialize mapper
        mapper = RedirectMapper(args.api_key, use_cache=not args.no_cache, batch_size=args.batch_size)

        # Extract URLs
        old_urls = mapper.extract_urls_from_file(args.old_urls)
        new_urls = mapper.extract_urls_from_file(args.new_urls)

        # Generate redirects
        redirects = mapper.generate_redirect_suggestions(old_urls, new_urls)

        # Export redirects
        mapper.export_redirects(redirects, args.output, args.format)

        # Generate report if requested
        if args.report:
            report = mapper.generate_analytics_report(redirects)
            report_file = Path(args.output).with_suffix('.report.json')
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📊 Analytics report saved to {report_file}")
            print(f"Average confidence: {report['avg_confidence']:.2f}")
            print(f"Total redirects: {report['total_redirects']}")
            print(f"Low confidence - Homepage fallbacks: {report['fallback_count']}")


if __name__ == "__main__":
    main()
