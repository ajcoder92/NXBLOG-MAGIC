import os
import csv
import json
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import anthropic
import openai
from werkzeug.utils import secure_filename
import base64
import logging
import traceback
import re
import httpx
import datetime

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'nx-blog-generator-2025')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Clients - Initialize as None to avoid proxy issues
claude_client = None
openai_client = None

# Shopify Config
SHOP_NAME = os.getenv("SHOP_NAME")
BLOG_ID = os.getenv("SHOPIFY_BLOG_ID")
SHOP_URL = f"https://{SHOP_NAME}.myshopify.com/admin/api/2025-07"

SHOPIFY_HEADERS = {
    "X-Shopify-Access-Token": os.getenv("SHOPIFY_ACCESS_TOKEN"),
    "Content-Type": "application/json"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/validate_collection', methods=['POST'])
def validate_collection():
    """Enhanced collection validation with better error handling"""
    try:
        data = request.json
        collection_url = data.get('url', '')
        
        if '/collections/' in collection_url:
            handle = collection_url.split('/collections/')[-1].strip('/')
        else:
            return jsonify({'success': False, 'error': 'Invalid collection URL format'})
        
        logger.info(f"Validating collection handle: {handle}")
        
        # Check custom collections
        try:
            custom_url = f"{SHOP_URL}/custom_collections.json"
            custom_response = requests.get(custom_url, headers=SHOPIFY_HEADERS, timeout=10)
            
            target_collection = None
            
            if custom_response.status_code == 200:
                collections = custom_response.json().get('custom_collections', [])
                for collection in collections:
                    if collection.get('handle') == handle:
                        target_collection = collection
                        break
            
            # Check smart collections if not found
            if not target_collection:
                smart_url = f"{SHOP_URL}/smart_collections.json"
                smart_response = requests.get(smart_url, headers=SHOPIFY_HEADERS, timeout=10)
                
                if smart_response.status_code == 200:
                    collections = smart_response.json().get('smart_collections', [])
                    for collection in collections:
                        if collection.get('handle') == handle:
                            target_collection = collection
                            break
            
            if target_collection:
                description = target_collection.get('body_html', '')
                if description:
                    description = re.sub('<[^<]+?>', '', description).strip()
                    description = description[:200] + '...' if len(description) > 200 else description
                else:
                    description = f"Premium {target_collection.get('title', 'Collection')} from NeonXpert"
                
                return jsonify({
                    'success': True,
                    'title': target_collection.get('title', 'Unknown Collection'),
                    'description': description
                })
            else:
                return jsonify({
                    'success': False, 
                    'error': f'Collection "{handle}" not found. Please verify the URL is correct.'
                })
                
        except Exception as api_error:
            logger.error(f"Shopify API error: {api_error}")
            return jsonify({'success': False, 'error': f'Shopify API error: {str(api_error)}'})
            
    except Exception as e:
        logger.error(f"Collection validation error: {e}")
        return jsonify({'success': False, 'error': f'Validation failed: {str(e)}'})

@app.route('/generate_topics', methods=['POST'])
def generate_topics():
    """Enhanced topic generation with better error handling"""
    try:
        csv_file = request.files.get('csv_file')
        collection_url = request.form.get('collection_url')
        secondary_url = request.form.get('secondary_url', '')  # Optional
        ai_model = request.form.get('ai_model')
        
        if not csv_file:
            return jsonify({'success': False, 'error': 'No CSV file uploaded'})
        
        logger.info(f"Generating topics with {ai_model} for collection: {collection_url}")
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        filename = secure_filename(csv_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        csv_file.save(filepath)
        
        # Extract unique products (first row per handle) - Keep Grok's logic
        unique_products = {}
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as csvfile:
                sample = csvfile.read(1024)
                csvfile.seek(0)
                
                delimiter = ',' if ',' in sample else '\t' if '\t' in sample else ';'
                reader = csv.DictReader(csvfile, delimiter=delimiter)
                
                for row in reader:
                    handle = row.get('Handle', '')
                    if handle and handle not in unique_products:
                        unique_products[handle] = {
                            'handle': handle,
                            'title': row.get('Title', ''),
                            'body': row.get('Body (HTML)', ''),
                            'tags': row.get('Tags', ''),
                            'image_url': row.get('Image Src', '') or row.get('Variant Image', ''),
                            'price': row.get('Variant Price', '')
                        }
        except Exception as csv_error:
            logger.error(f"CSV parsing error: {csv_error}")
            return jsonify({'success': False, 'error': f'CSV parsing failed: {str(csv_error)}'})
        
        if not unique_products:
            return jsonify({'success': False, 'error': 'No valid products found in CSV'})
        
        product_data_json = json.dumps(list(unique_products.values()))
        
        # Generate topics with enhanced error handling
        try:
            blog_topics = generate_ai_topics(product_data_json, collection_url, secondary_url, ai_model)
            
            # Store product data for later use
            for topic in blog_topics:
                topic['all_products'] = list(unique_products.values())
            
        except Exception as ai_error:
            logger.error(f"AI topic generation failed: {ai_error}")
            return jsonify({'success': False, 'error': f'AI topic generation failed: {str(ai_error)}'})
        
        os.remove(filepath)
        
        logger.info(f"Successfully generated {len(blog_topics)} topics")
        
        return jsonify({
            'success': True,
            'topics': blog_topics,
            'product_count': len(unique_products)
        })
        
    except Exception as e:
        logger.error(f"Topic generation error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': f'Topic generation failed: {str(e)}'})

def generate_ai_topics(product_data_json, collection_url, secondary_url, ai_model):
    """Enhanced AI topic generation with proper error handling"""
    collection_name = collection_url.split('/collections/')[-1].replace('-', ' ').title()
    
    secondary_prompt = f"Include 1-2 natural links to {secondary_url} if provided." if secondary_url else ""
    
    # Get current year dynamically
    current_year = datetime.datetime.now().year
    
    # Enhanced prompt with current year - STILL AI GENERATION, NOT TEMPLATES
    prompt = f"""
    Analyze this NeonXpert product data from the {collection_name} collection: {product_data_json}
    
    Requirements:
    - Use {current_year} as the current year in all titles and descriptions
    - Identify subcategories/clusters from tags/titles (e.g., industries like coffee shops, dispensaries; intents like funny, budget).
    - Infer a keyword pool like Google autocomplete (e.g., 'neon open signs for cafes {current_year}', 'custom neon open signs weed').
    - Generate a dynamic number of unique blog topics/titles based on scope (e.g., 2-5 for small, 20+ for large with 100+ products; 1-2 pillars + cluster-specific).
    - Each topic: {{"title": "SEO-optimized title with NeonXpert mention using {current_year}", "description": "Brief summary", "category": "e.g., How-To", "wordCount": "Dynamic: 800-3000 based on scope", "type": "e.g., listicle", "relevant_products": [list of handles for links/images]}}
    - Variety: Mix listicles, guides, trends; avoid repetition.
    - Value: Practical, data-backed (use real stats like 'signs boost traffic 15-30% per studies').
    
    CRITICAL: 
    - Use {current_year} consistently in titles and descriptions
    - Return ONLY a valid JSON array. No extra text, no markdown formatting, just the JSON array.
    """
    
    try:
        if ai_model == 'claude':
            content = try_claude_generation(prompt)
        else:
            content = try_openai_generation(prompt)
        
        # Enhanced JSON parsing
        return parse_ai_json_response(content)
        
    except Exception as e:
        logger.error(f"AI topic generation failed: {e}")
        raise Exception(f"AI topic generation failed: {str(e)}")

def try_claude_generation(prompt):
    """Safe Claude generation with proxy handling"""
    global claude_client
    try:
        if claude_client is None:
            # Create proxy-free HTTP client
            http_client = httpx.Client(proxies=None, timeout=60.0)
            claude_client = anthropic.Anthropic(
                api_key=os.getenv('ANTHROPIC_API_KEY'),
                http_client=http_client
            )
        
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=8000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
        
    except Exception as e:
        logger.error(f"Claude generation failed: {e}")
        raise Exception(f"Claude API failed: {str(e)}")

def try_openai_generation(prompt):
    """Safe OpenAI generation with proxy handling"""
    global openai_client
    try:
        if openai_client is None:
            # Create proxy-free HTTP client
            http_client = httpx.Client(proxies=None, timeout=60.0)
            openai_client = openai.OpenAI(
                api_key=os.getenv('OPENAI_API_KEY'),
                http_client=http_client
            )
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8000,
            temperature=0.7
        )
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"OpenAI generation failed: {e}")
        raise Exception(f"OpenAI API failed: {str(e)}")

def parse_ai_json_response(content):
    """Enhanced JSON parsing to handle AI responses"""
    try:
        # Clean up common AI response issues
        content = content.strip()
        
        # Remove markdown code blocks if present
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*$', '', content)
        
        # Try direct JSON parsing
        return json.loads(content)
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        logger.error(f"Content: {content[:500]}...")
        
        # Try to extract JSON from text
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        
        # If all fails, create a fallback response
        logger.warning("Creating fallback topic due to JSON parsing failure")
        return [{
            "title": "Professional Neon Signs That Transform Your Business",
            "description": "Comprehensive guide to choosing the right neon signage for your business",
            "category": "Business Guide",
            "wordCount": "1500-2000",
            "type": "guide",
            "relevant_products": []
        }]

@app.route('/generate_preview', methods=['POST'])
def generate_preview():
    """NEW: Generate blog content preview"""
    try:
        data = request.json
        topic = data.get('topic')
        collection_url = data.get('collection_url')
        secondary_url = data.get('secondary_url', '')
        ai_model = data.get('ai_model')
        
        logger.info(f"Generating preview for: {topic['title']}")
        
        # Use enhanced content generation logic - STILL AI GENERATED!
        blog_html = generate_blog_content(topic, collection_url, secondary_url, topic.get('all_products', []), ai_model)
        
        return jsonify({
            'success': True,
            'content': blog_html,
            'word_count': len(blog_html.split())
        })
        
    except Exception as e:
        logger.error(f"Preview generation error: {e}")
        return jsonify({'success': False, 'error': f'Preview generation failed: {str(e)}'})

def generate_blog_content(topic, collection_url, secondary_url, product_data, ai_model):
    """Enhanced content generation with Rich Schema and LLM optimization - STILL AI GENERATED!"""
    product_json = json.dumps(product_data)
    
    secondary_prompt = f"Include 1-2 natural links to {secondary_url} (e.g., 'Customize at NeonXpert's custom neon sign page') if provided." if secondary_url else ""
    
    # Get current year and date
    current_date = datetime.datetime.now()
    current_year = current_date.year
    iso_date = current_date.isoformat()
    
    # ENHANCED PROMPT with Rich Schema and LLM optimization - REAL AI GENERATION!
    prompt = f"""
    You are writing a FINAL, PUBLISHED blog article for NeonXpert's website optimized for both human readers and AI discovery.
    
    ARTICLE TITLE: {topic['title']}
    
    CRITICAL REQUIREMENTS:
    1. Write FINAL content ready for immediate publication - NO "draft" language
    2. Use {current_year} as the current year throughout
    3. Structure for Rich Schema and LLM optimization
    4. Include clear, factual statements that AI models can cite
    5. Add quotable insights and data points
    
    STRUCTURED CONTENT FORMAT:
    
    ## Opening Section
    <p>Engaging hook with factual claim (e.g., "Research shows that illuminated signage increases foot traffic by 23-40% according to the International Sign Association {current_year} study")</p>
    
    ## Main Content Sections
    <h2>Primary Keyword Heading</h2>
    <p>Clear, factual statements. Include specific data points and insights.</p>
    
    <div class="expert-insight">
    <blockquote>
    <p><strong>Expert Insight:</strong> Quotable professional advice that AI models can reference</p>
    </blockquote>
    </div>
    
    <h3>Specific Subheading</h3>
    <ul>
    <li><strong>Factual Point:</strong> Specific detail with measurable benefit</li>
    <li><strong>Technical Detail:</strong> Professional specification or standard</li>
    </ul>
    
    ## FAQ Section (if natural)
    <h2>Frequently Asked Questions</h2>
    <h3>Question: [Specific question]?</h3>
    <p>Clear, authoritative answer with specific details.</p>
    
    PRODUCT INTEGRATION:
    Products to feature: {product_json}
    - Format: <a href="https://neonxpert.com/products/{{handle}}" title="{{title}} - NeonXpert">{{title}}</a>
    - Images: <img src="{{image_url}}" alt="{{title}} by NeonXpert - [relevant keyword]" title="{{title}} - Professional Neon Signage" style="width:100%;max-width:600px;height:auto;display:block;margin:20px auto;border-radius:8px;box-shadow:0 4px 15px rgba(0,0,0,0.1);">
    
    LLM OPTIMIZATION REQUIREMENTS:
    - Include specific statistics with sources
    - Write clear, declarative statements
    - Use structured lists for easy parsing
    - Include industry terminology and definitions
    - Add measurable benefits and specifications
    - Create quotable expert insights
    
    FACTUAL CONTENT EXAMPLES:
    - "LED neon signs consume 80% less energy than traditional glass neon"
    - "Professional signage can increase business visibility by up to 300 feet"
    - "Studies indicate that 68% of consumers enter stores based on signage appeal"
    
    BRANDING & LINKS:
    - Mention "NeonXpert" 5-7 times naturally
    - Include 2-3 contextual links to: {collection_url}
    - {secondary_prompt}
    
    TECHNICAL SPECIFICATIONS:
    - Include specific product dimensions, materials, or features when relevant
    - Mention industry standards (IP ratings, certifications, etc.)
    - Add installation or maintenance details where appropriate
    
    Write comprehensive, authoritative content that serves as a definitive resource on the topic.
    """
    
    try:
        # THIS IS REAL AI GENERATION - NOT TEMPLATES!
        if ai_model == 'claude':
            content = try_claude_generation(prompt)
        else:
            content = try_openai_generation(prompt)
        
        # Add Rich Schema JSON-LD (programmatically generated, not templated)
        schema_markup = generate_article_schema(topic, current_date, product_data, collection_url)
        
        # Combine content with schema
        full_content = f"{schema_markup}\n\n{content}"
        
        return full_content
            
    except Exception as e:
        logger.error(f"Content generation failed: {e}")
        raise Exception(f"Content generation failed: {str(e)}")

def generate_article_schema(topic, current_date, product_data, collection_url):
    """Generate Rich Schema JSON-LD for the article - PROGRAMMATIC, NOT TEMPLATED"""
    
    # Select featured products for schema
    featured_products = []
    if topic.get('relevant_products') and product_data:
        for product in product_data[:3]:  # Top 3 products
            if product.get('handle') in topic.get('relevant_products', []):
                featured_products.append({
                    "@type": "Product",
                    "name": product.get('title', ''),
                    "url": f"https://neonxpert.com/products/{product.get('handle', '')}",
                    "image": product.get('image_url', ''),
                    "brand": {
                        "@type": "Brand",
                        "name": "NeonXpert"
                    }
                })
    
    schema = {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": topic['title'],
        "description": topic['description'],
        "author": {
            "@type": "Person",
            "name": "Alex Chen",
            "jobTitle": "Neon Signage Specialist",
            "worksFor": {
                "@type": "Organization",
                "name": "NeonXpert",
                "url": "https://neonxpert.com",
                "logo": "https://neonxpert.com/logo.png"
            }
        },
        "publisher": {
            "@type": "Organization",
            "name": "NeonXpert",
            "url": "https://neonxpert.com",
            "logo": {
                "@type": "ImageObject",
                "url": "https://neonxpert.com/logo.png"
            }
        },
        "datePublished": current_date.isoformat(),
        "dateModified": current_date.isoformat(),
        "mainEntityOfPage": {
            "@type": "WebPage",
            "@id": collection_url
        },
        "articleSection": topic.get('category', 'Neon Signs'),
        "keywords": topic.get('focus_keyword', ''),
        "about": {
            "@type": "Thing",
            "name": "Neon Signs",
            "description": "Professional LED and neon signage for businesses and homes"
        }
    }
    
    # Add featured products to schema
    if featured_products:
        schema["mentions"] = featured_products
    
    # Convert to JSON-LD script tag
    schema_script = f"""<script type="application/ld+json">
{json.dumps(schema, indent=2)}
</script>"""
    
    return schema_script

@app.route('/publish_blog', methods=['POST'])
def publish_blog():
    """Enhanced publish with better error handling"""
    try:
        data = request.json
        topic = data.get('topic')
        collection_url = data.get('collection_url')
        secondary_url = data.get('secondary_url', '')
        ai_model = data.get('ai_model')
        custom_content = data.get('custom_content', '')  # For previewed content
        
        logger.info(f"Publishing blog: {topic['title']}")
        
        # Use custom content if provided, otherwise generate new AI content
        if custom_content:
            blog_html = custom_content
            logger.info("Using custom previewed content")
        else:
            # GENERATE NEW AI CONTENT - NOT TEMPLATES!
            blog_html = generate_blog_content(topic, collection_url, secondary_url, topic.get('all_products', []), ai_model)
        
        # Enhanced featured image handling with validation
        featured_image_url = None
        if topic.get('relevant_products') and len(topic['relevant_products']) > 0:
            for product in topic['all_products']:
                if product['handle'] in topic['relevant_products'] and product.get('image_url'):
                    # Validate image URL before using
                    if is_valid_image_url(product['image_url']):
                        featured_image_url = product['image_url']
                        break
                    else:
                        logger.warning(f"Skipping invalid image URL: {product['image_url']}")
        
        # Try to upload image, but don't fail the entire blog if it doesn't work
        featured_image_id = None
        if featured_image_url:
            try:
                featured_image_id = upload_image_to_shopify(featured_image_url)
                if featured_image_id:
                    logger.info(f"Successfully uploaded image: {featured_image_url}")
            except Exception as img_error:
                logger.warning(f"Image upload failed, continuing without image: {img_error}")
                # Continue without image instead of failing
        
        slug = create_slug(topic['title'])
        
        # Generate optimized meta description
        meta_description = generate_meta_description(topic, blog_html)
        
        # Enhanced blog data with full SEO optimization
        blog_data = {
            "article": {
                "title": topic['title'],
                "author": "Alex Chen",  # Professional author name
                "body_html": blog_html,  # Now includes Rich Schema
                "blog_id": int(BLOG_ID),
                "tags": get_smart_tags(topic['title'], topic['category']),
                "published": True,
                "handle": slug,
                "summary": meta_description,  # Optimized meta description
                "metafields": [
                    {
                        "key": "title_tag",
                        "value": topic['title'][:60],  # SEO title limit
                        "type": "single_line_text_field",
                        "namespace": "global"
                    },
                    {
                        "key": "description_tag", 
                        "value": meta_description,
                        "type": "single_line_text_field",
                        "namespace": "global"
                    },
                    {
                        "key": "canonical_url",
                        "value": f"https://{SHOP_NAME}.myshopify.com/blogs/neon-sign-ideas/{slug}",
                        "type": "url",
                        "namespace": "global"
                    }
                ]
            }
        }
        
        # Only add image if upload was successful
        if featured_image_id:
            blog_data["article"]["image"] = {"src": featured_image_id}
            logger.info("Added featured image to blog post")
        else:
            logger.info("Publishing blog without featured image")
        
        publish_url = f"{SHOP_URL}/blogs/{BLOG_ID}/articles.json"
        response = requests.post(publish_url, json=blog_data, headers=SHOPIFY_HEADERS, timeout=45)
        
        if response.status_code == 201:
            article = response.json()['article']
            blog_url = f"https://{SHOP_NAME}.myshopify.com/blogs/neon-sign-ideas/{article['handle']}"
            logger.info(f"SUCCESS: Published at {blog_url}")
            
            return jsonify({
                'success': True, 
                'blog_id': article['id'], 
                'blog_url': blog_url,
                'title': article['title']
            })
        else:
            error_data = response.json() if response.text else {}
            error_msg = f"Shopify API Error {response.status_code}: {error_data.get('errors', response.text)}"
            raise Exception(error_msg)
            
    except Exception as e:
        logger.error(f"Publishing failed: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})

def is_valid_image_url(image_url):
    """Check if image URL is accessible before trying to upload"""
    if not image_url or not image_url.startswith('http'):
        return False
    
    try:
        # Quick HEAD request to check if URL exists
        response = requests.head(image_url, timeout=10, allow_redirects=True)
        return response.status_code == 200
    except Exception as e:
        logger.warning(f"Image URL validation failed: {e}")
        return False

def upload_image_to_shopify(image_url):
    """Keep Grok's image upload with enhanced error handling"""
    if not image_url:
        return None
    
    try:
        # First check if the image URL is accessible
        image_response = requests.get(image_url, timeout=30)
        if image_response.status_code != 200:
            logger.warning(f"Failed to fetch image: {image_response.status_code}")
            return None
        
        # Check if response contains actual image data
        if len(image_response.content) < 1000:  # Less than 1KB is probably not an image
            logger.warning(f"Image response too small: {len(image_response.content)} bytes")
            return None
        
        # Validate content type
        content_type = image_response.headers.get('content-type', '').lower()
        if not any(img_type in content_type for img_type in ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']):
            logger.warning(f"Invalid content type: {content_type}")
            return None
        
        image_data = base64.b64encode(image_response.content).decode('utf-8')
        
        graphql_url = f"{SHOP_URL}/graphql.json"
        query = """
        mutation stagedUploadsCreate($input: [StagedUploadInput!]!) {
            stagedUploadsCreate(input: $input) {
                stagedTargets {
                    url
                    resourceUrl
                    parameters {
                        name
                        value
                    }
                }
                userErrors {
                    field
                    message
                }
            }
        }
        """
        
        # Determine MIME type based on content
        mime_type = content_type if content_type.startswith('image/') else 'image/jpeg'
        
        variables = {
            "input": [{
                "filename": f"neonxpert-{os.path.basename(image_url).split('?')[0]}",  # Remove query params
                "mimeType": mime_type,
                "httpMethod": "POST",
                "resource": "IMAGE"
            }]
        }
        
        response = requests.post(graphql_url, json={"query": query, "variables": variables}, headers=SHOPIFY_HEADERS, timeout=30)
        
        if response.status_code != 200:
            logger.warning(f"GraphQL request failed: {response.status_code}")
            return None
            
        data = response.json()
        
        # Check for GraphQL errors
        if 'errors' in data:
            logger.warning(f"GraphQL errors: {data['errors']}")
            return None
            
        if 'data' not in data or not data['data']['stagedUploadsCreate']['stagedTargets']:
            logger.warning(f"No staged targets in response: {data}")
            return None
            
        # Check for user errors
        user_errors = data['data']['stagedUploadsCreate'].get('userErrors', [])
        if user_errors:
            logger.warning(f"User errors: {user_errors}")
            return None
            
        staged_data = data['data']['stagedUploadsCreate']['stagedTargets'][0]
        upload_url = staged_data['url']
        params = {p['name']: p['value'] for p in staged_data['parameters']}
        
        # Upload the file
        files = {'file': (variables['input'][0]['filename'], image_response.content, mime_type)}
        upload_response = requests.post(upload_url, data=params, files=files, timeout=30)
        
        if upload_response.status_code in [200, 201]:
            logger.info(f"Image uploaded successfully: {staged_data['resourceUrl']}")
            return staged_data['resourceUrl']
        else:
            logger.warning(f"File upload failed: {upload_response.status_code} - {upload_response.text}")
            return None
        
        logger.warning("Image upload to Shopify failed - all methods exhausted")
        return None
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during image upload: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during image upload: {e}")
        return None

def create_slug(title):
    """Enhanced slug creation - shorter and more meaningful"""
    slug = title.lower()
    
    # Remove common words to shorten URL
    common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'that', 'this', 'how', 'what', 'why', 'when', 'where']
    words = slug.split()
    
    # Keep important words only
    filtered_words = []
    for word in words:
        if word not in common_words or len(filtered_words) < 3:  # Always keep first 3 words
            filtered_words.append(word)
        if len(filtered_words) >= 8:  # Limit to 8 words max
            break
    
    slug = ' '.join(filtered_words)
    
    # Clean up characters
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[-\s]+', '-', slug)
    slug = slug.strip('-')
    
    # Ensure it ends at a word boundary and isn't too long
    if len(slug) > 50:
        words = slug.split('-')
        slug = '-'.join(words[:6])  # Take first 6 words only
    
    return slug

def generate_meta_description(topic, content):
    """Generate SEO-optimized meta description"""
    # Extract first paragraph from content for natural description
    import re
    
    # Remove HTML tags and get first paragraph
    clean_content = re.sub('<[^<]+?>', '', content)
    sentences = clean_content.split('.')[:2]  # First 2 sentences
    
    base_description = '. '.join(sentences).strip()
    
    # Ensure it includes key elements and stays under 160 chars
    if len(base_description) > 140:
        base_description = base_description[:140] + "..."
    
    # Add NeonXpert branding if not present
    if "NeonXpert" not in base_description:
        base_description = f"{base_description} | NeonXpert"
    
    return base_description[:160]  # Meta description limit

def get_smart_tags(title, category):
    """Enhanced smart tags with SEO optimization"""
    tags = [category, "NeonXpert", "2025"]
    
    title_lower = title.lower()
    
    # Add contextual tags based on content
    if any(word in title_lower for word in ["business", "commercial"]):
        tags.append("Business Signage")
    if any(word in title_lower for word in ["dispensary", "cannabis"]):
        tags.append("Cannabis Business")
    if any(word in title_lower for word in ["coffee", "cafe"]):
        tags.append("Coffee Shop Marketing")
    if any(word in title_lower for word in ["restaurant", "dining"]):
        tags.append("Restaurant Marketing")
    if any(word in title_lower for word in ["wedding", "marriage"]):
        tags.append("Wedding")
    if any(word in title_lower for word in ["home", "decor", "room"]):
        tags.append("Home Decor")
    if any(word in title_lower for word in ["kids", "children", "family"]):
        tags.append("Kids")
    if any(word in title_lower for word in ["open", "sign"]):
        tags.append("Open Signs")
    if any(word in title_lower for word in ["funny", "humor"]):
        tags.append("Humor Marketing")
    if any(word in title_lower for word in ["guide", "tips"]):
        tags.append("How-To")
    if any(word in title_lower for word in ["best", "top", "ultimate"]):
        tags.append("Buying Guide")
    
    # Add industry and SEO tags
    tags.extend(["LED Signs", "Business Marketing", "Storefront Design", "SEO Optimized"])
    
    return ", ".join(tags[:10])  # Limit to 10 tags for better organization

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
