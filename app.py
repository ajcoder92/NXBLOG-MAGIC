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
        
        # Extract unique products (ONLY first row per handle with actual data)
        unique_products = {}
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as csvfile:
                sample = csvfile.read(1024)
                csvfile.seek(0)
                
                delimiter = ',' if ',' in sample else '\t' if '\t' in sample else ';'
                reader = csv.DictReader(csvfile, delimiter=delimiter)
                
                for row in reader:
                    handle = row.get('Handle', '').strip()
                    title = row.get('Title', '').strip()
                    
                    # STRICT FILTERING: Only process rows with both handle AND title
                    # This eliminates variant rows that have handle but empty title
                    if handle and title and handle not in unique_products:
                        body = row.get('Body (HTML)', '').strip()
                        tags = row.get('Tags', '').strip()
                        
                        # Skip if this looks like a variant row (has handle but no meaningful content)
                        if len(title) < 5:  # Skip very short/empty titles
                            continue
                            
                        unique_products[handle] = {
                            'handle': handle,
                            'title': title,
                            'body': body,  # This is the description
                            'tags': tags,
                            'image_url': row.get('Image Src', '').strip() or row.get('Variant Image', '').strip(),
                            'price': row.get('Variant Price', ''),
                            'theme': row.get('Theme (product.metafields.shopify.theme)', '').strip()
                        }
        except Exception as csv_error:
            logger.error(f"CSV parsing error: {csv_error}")
            return jsonify({'success': False, 'error': f'CSV parsing failed: {str(csv_error)}'})
        
        if not unique_products:
            return jsonify({'success': False, 'error': 'No valid products found in CSV'})
        
        # Debug logging to see what products were extracted
        logger.info(f"Extracted {len(unique_products)} unique products:")
        for handle, product in list(unique_products.items())[:5]:  # Log first 5 products
            logger.info(f"- {handle}: {product['title'][:50]}...")
        
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
    
    # Enhanced prompt focused on ACTUAL product analysis
    prompt = f"""
    Analyze this REAL NeonXpert product data from the {collection_name} collection: {product_data_json}
    
    TASK: Create blog topics based on the ACTUAL products provided.
    
    ANALYSIS REQUIREMENTS:
    1. Look at the actual product TITLES and DESCRIPTIONS in the data
    2. Identify common THEMES from the product names (e.g., if you see "Rainbow Heart", "Pride Flag", "Love Wins" - the theme is LGBTQ pride)
    3. Group products by their actual characteristics (colors, designs, messages, room types)
    4. Create topics that would help customers choose between these SPECIFIC products
    
    TOPIC CREATION RULES:
    1. Use {current_year} as the current year in all titles
    2. Base topics on the actual product names and themes you see in the data
    3. Create 3-8 topics depending on product variety (more products = more topics)
    4. Focus on helping customers understand THESE specific products
    5. Each topic should feature 3-5 specific products from the data
    
    TOPIC FORMAT:
    {{"title": "Descriptive title using {current_year} that helps customers choose from your actual products", "description": "Brief summary of what this topic covers", "category": "Guide/Showcase/Comparison", "wordCount": "1200-2000", "type": "guide/showcase/comparison", "relevant_products": [list of 3-5 actual product handles from the data]}}
    
    EXAMPLE ANALYSIS:
    If your data contains products like "Rainbow Heart Neon Sign", "Pride Flag LED Display", "Love Wins Neon Art":
    - Theme: LGBTQ Pride celebration
    - Possible topics: "Best LGBTQ Pride Neon Signs for Home Decor in {current_year}", "Rainbow vs Multi-Color: Choosing Pride Neon Signs", "Small vs Large: Pride Neon Signs for Every Space"
    
    CRITICAL: 
    - Analyze the ACTUAL product titles in the data provided
    - Create topics that showcase THESE specific products
    - Use {current_year} in titles
    - Return ONLY a valid JSON array, no extra text
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
    """Safe Claude generation with auto-fallback to OpenAI on overload"""
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
        error_str = str(e)
        logger.error(f"Claude generation failed: {error_str}")
        
        # Auto-fallback to OpenAI if Claude is overloaded (529 error)
        if "529" in error_str or "overloaded" in error_str.lower():
            logger.info("Claude overloaded, auto-falling back to OpenAI...")
            try:
                return try_openai_generation(prompt)
            except Exception as fallback_error:
                logger.error(f"OpenAI fallback also failed: {fallback_error}")
                raise Exception(f"Both Claude (overloaded) and OpenAI failed: {str(fallback_error)}")
        
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
            "title": f"Professional Neon Signs That Transform Your Business {datetime.datetime.now().year}",
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
    """Enhanced content generation with HubSpot-style aesthetics - NO SCHEMA MARKUP!"""
    product_json = json.dumps(product_data)
    
    secondary_prompt = f"Include 1-2 natural links to {secondary_url} (e.g., 'Customize at NeonXpert's custom neon sign page') if provided." if secondary_url else ""
    
    # Get current year and date
    current_date = datetime.datetime.now()
    current_year = current_date.year
    
    # CLEAN PROMPT - NO SCHEMA, NO BROKEN TEXT
    prompt = f"""
    You are writing a FINAL, PUBLISHED blog article for NeonXpert's website about their ACTUAL products.
    
    ARTICLE TITLE: {topic['title']}
    
    REAL PRODUCT DATA TO FEATURE: {product_json}
    
    CRITICAL FORMATTING REQUIREMENTS:
    1. Write FINAL content ready for immediate publication - NO "draft" language  
    2. Use {current_year} as the current year throughout
    3. Focus ONLY on the actual products provided - their titles, descriptions, themes
    4. NO FAKE STATISTICS - Do not make up percentages, studies, or data
    5. NO FAKE EXPERT QUOTES - Do not quote fictional experts
    6. OUTPUT ONLY CLEAN HTML - No schema markup, no leftover template text
    7. INCLUDE ALL INLINE STYLES - Every element must have complete styling
    8. NO BROKEN TEXT - Do not include partial sentences or template artifacts
    
    CONTENT STRUCTURE - HUBSPOT-STYLE AESTHETICS:
    
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 12px; color: white; margin: 20px 0;">
    <h1 style="color: white; font-size: 28px; margin-bottom: 15px; text-align: center; line-height: 1.3;">{topic['title']}</h1>
    <p style="font-size: 18px; text-align: center; margin: 0; opacity: 0.9;">Engaging introduction about the specific products, focusing on their actual features and benefits.</p>
    </div>
    
    <div style="background: #f8f9ff; border-left: 4px solid #667eea; padding: 20px; margin: 25px 0; border-radius: 8px;">
    <h3 style="color: #667eea; margin: 0 0 10px 0; font-size: 18px;"><i>💡 Key Takeaway</i></h3>
    <p style="margin: 0; font-weight: 500; line-height: 1.5;">Brief summary of what customers will learn from this article.</p>
    </div>
    
    <h2 style="color: #2d3748; font-size: 24px; margin: 30px 0 20px 0; padding-bottom: 10px; border-bottom: 2px solid #667eea;">Featured Products from NeonXpert</h2>
    
    For each product, create visually appealing sections:
    
    <div style="background: white; border-radius: 12px; padding: 25px; margin: 25px 0; box-shadow: 0 4px 20px rgba(0,0,0,0.1); border: 1px solid #e2e8f0;">
    <h3 style="color: #2d3748; font-size: 20px; margin: 0 0 15px 0;">[Actual Product Title]</h3>
    <div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: center;">
    <div style="flex: 1; min-width: 300px;">
    <p style="line-height: 1.6; color: #4a5568; margin-bottom: 15px;">Description based on actual product information. Focus on design, colors, themes, and practical uses.</p>
    <div style="background: #e6fffa; padding: 15px; border-radius: 8px; margin: 15px 0;">
    <p style="margin: 0; color: #065f46; font-weight: 500;"><strong>Perfect For:</strong> Specific use cases based on product features</p>
    </div>
    </div>
    <div style="flex: 1; min-width: 250px; text-align: center;">
    <img src="{{image_url}}" alt="{{title}} by NeonXpert" title="{{title}} - Professional Neon Signage" style="width:100%;max-width:400px;height:auto;border-radius:12px;box-shadow:0 8px 25px rgba(0,0,0,0.15); margin-bottom: 15px;">
    <a href="https://neonxpert.com/products/{{handle}}" style="display: inline-block; background: #667eea; color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: 600; transition: all 0.3s ease; cursor: pointer;">Shop Now</a>
    </div>
    </div>
    </div>
    
    <div style="background: #fffbeb; border: 1px solid #f59e0b; border-radius: 12px; padding: 25px; margin: 30px 0;">
    <h2 style="color: #92400e; font-size: 22px; margin: 0 0 20px 0;"><i>✨</i> Styling and Placement Ideas</h2>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
    <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
    <h4 style="color: #92400e; margin: 0 0 10px 0; font-size: 16px;">Room Placement</h4>
    <p style="margin: 0; font-size: 14px; line-height: 1.5; color: #4a5568;">Specific suggestions based on actual product themes</p>
    </div>
    <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
    <h4 style="color: #92400e; margin: 0 0 10px 0; font-size: 16px;">Design Tips</h4>
    <p style="margin: 0; font-size: 14px; line-height: 1.5; color: #4a5568;">Styling tips that match actual product characteristics</p>
    </div>
    </div>
    </div>
    
    <div style="background: #f0f9ff; border-radius: 12px; padding: 25px; margin: 30px 0;">
    <h2 style="color: #0369a1; font-size: 22px; margin: 0 0 20px 0;"><i>🔧</i> Installation and Care</h2>
    <div style="background: white; padding: 20px; border-radius: 8px; border-left: 4px solid #0369a1;">
    <p style="margin: 0; line-height: 1.6; color: #4a5568;">General information about LED neon sign installation and maintenance. Keep factual and practical.</p>
    </div>
    </div>
    
    PRODUCT INTEGRATION REQUIREMENTS:
    - Feature each product naturally in the content
    - Use actual product titles and descriptions from the data provided
    - Link format: <a href="https://neonxpert.com/products/{{handle}}" title="{{title}} - NeonXpert">{{title}}</a>
    - Include product images with proper alt text
    
    CONTENT QUALITY REQUIREMENTS:
    - Only mention general LED neon benefits (energy efficiency, longevity, safety)
    - Do not invent specific statistics or studies  
    - Do not quote experts or authorities
    - Focus on the actual product features and visual appeal
    - Write from the perspective of showcasing real products
    
    VISUAL REQUIREMENTS:
    - Use the exact HTML structure and styling provided above
    - Every section must have proper background colors, padding, and styling
    - Include all gradient backgrounds, shadows, and visual elements
    - Make it look like a premium, professional blog post
    - Ensure responsive design with proper flex layouts
    - NO broken text, NO template artifacts, NO partial sentences
    
    BRANDING & LINKS:
    - Mention "NeonXpert" naturally 3-5 times
    - Include 2-3 contextual links to: {collection_url}
    - {secondary_prompt}
    
    CRITICAL: Write engaging, informative content using the EXACT HTML structure above. Output ONLY the styled HTML content - no schema markup, no broken text, no template artifacts. The final output should be clean, professional, and ready for immediate publication.
    """
    
    try:
        # THIS IS REAL AI GENERATION - NOT TEMPLATES!
        if ai_model == 'claude':
            content = try_claude_generation(prompt)
        else:
            content = try_openai_generation(prompt)
        
        # Clean up any broken HTML or template artifacts
        content = clean_content_artifacts(content)
        
        # NO SCHEMA MARKUP - user uses TinyIMG for that
        # Just return clean content
        return content
            
    except Exception as e:
        logger.error(f"Content generation failed: {e}")
        raise Exception(f"Content generation failed: {str(e)}")

# Schema generation function removed - user uses TinyIMG for schema markup

def clean_content_artifacts(content):
    """Clean up broken HTML, template artifacts, and formatting issues"""
    try:
        # Remove common template artifacts and broken text
        content = re.sub(r'```\s*

@app.route('/publish_blog', methods=['POST'])
def publish_blog():
    """Enhanced publish with better error handling and consistent content"""
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
        
        # Clean title for metafields (ensure single line)
        clean_title = re.sub(r'[\r\n\t]', ' ', topic['title'])
        clean_title = re.sub(r'\s+', ' ', clean_title).strip()[:60]
        
        # Enhanced blog data with full SEO optimization
        blog_data = {
            "article": {
                "title": topic['title'],
                "author": "NeonXpert Team",  # Professional team attribution
                "body_html": blog_html,  # Now includes Rich Schema
                "blog_id": int(BLOG_ID),
                "tags": get_smart_tags(topic['title'], topic['category']),
                "published": True,
                "handle": slug,
                "summary": meta_description,  # Optimized meta description
            }
        }
        
        # Add metafields only if we have valid values (Shopify is picky about these)
        try:
            if clean_title and meta_description:
                blog_data["article"]["metafields"] = [
                    {
                        "key": "title_tag",
                        "value": str(clean_title),  # Ensure string
                        "type": "single_line_text_field",
                        "namespace": "global"
                    },
                    {
                        "key": "description_tag", 
                        "value": str(meta_description),  # Ensure string
                        "type": "single_line_text_field",
                        "namespace": "global"
                    }
                ]
        except Exception as meta_error:
            logger.warning(f"Skipping metafields due to error: {meta_error}")
            # Continue without metafields if they cause issues
        
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
    """Generate SEO-optimized meta description - SINGLE LINE ONLY"""
    import re
    
    # Remove HTML tags and get first paragraph
    clean_content = re.sub('<[^<]+?>', '', content)
    
    # Remove all line breaks and extra spaces
    clean_content = re.sub(r'\s+', ' ', clean_content).strip()
    
    sentences = clean_content.split('.')[:2]  # First 2 sentences
    base_description = '. '.join(sentences).strip()
    
    # Ensure it includes key elements and stays under 160 chars
    if len(base_description) > 140:
        base_description = base_description[:140] + "..."
    
    # Add NeonXpert branding if not present
    if "NeonXpert" not in base_description:
        base_description = f"{base_description} | NeonXpert"
    
    # CRITICAL: Ensure single line and remove any problematic characters
    final_description = re.sub(r'[\r\n\t]', ' ', base_description)
    final_description = re.sub(r'\s+', ' ', final_description).strip()
    
    return final_description[:160]  # Meta description limit

def get_smart_tags(title, category):
    """Enhanced smart tags with SEO optimization"""
    tags = [category, "NeonXpert", str(datetime.datetime.now().year)]
    
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
    if any(word in title_lower for word in ["lgbtq", "pride", "rainbow"]):
        tags.append("LGBTQ Pride")
    
    # Add industry and SEO tags
    tags.extend(["LED Signs", "Business Marketing", "Storefront Design", "SEO Optimized"])
    
    return ", ".join(tags[:10])  # Limit to 10 tags for better organization

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port), '', content)  # Remove trailing ```
        content = re.sub(r'```[a-z]*\s*', '', content)  # Remove code block markers
        content = re.sub(r'Explore more.*?by NeonXpert\s*```?', '', content, flags=re.IGNORECASE)  # Remove broken explore text
        
        # Fix spacing issues around lists and elements
        content = re.sub(r'<div\s+style="[^"]*display:\s*grid[^>]*>\s*<div', r'<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;"><div', content)
        
        # Remove any incomplete HTML elements or partial sentences at end
        content = re.sub(r'<[^>]*

@app.route('/publish_blog', methods=['POST'])
def publish_blog():
    """Enhanced publish with better error handling and consistent content"""
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
        
        # Clean title for metafields (ensure single line)
        clean_title = re.sub(r'[\r\n\t]', ' ', topic['title'])
        clean_title = re.sub(r'\s+', ' ', clean_title).strip()[:60]
        
        # Enhanced blog data with full SEO optimization
        blog_data = {
            "article": {
                "title": topic['title'],
                "author": "NeonXpert Team",  # Professional team attribution
                "body_html": blog_html,  # Now includes Rich Schema
                "blog_id": int(BLOG_ID),
                "tags": get_smart_tags(topic['title'], topic['category']),
                "published": True,
                "handle": slug,
                "summary": meta_description,  # Optimized meta description
            }
        }
        
        # Add metafields only if we have valid values (Shopify is picky about these)
        try:
            if clean_title and meta_description:
                blog_data["article"]["metafields"] = [
                    {
                        "key": "title_tag",
                        "value": str(clean_title),  # Ensure string
                        "type": "single_line_text_field",
                        "namespace": "global"
                    },
                    {
                        "key": "description_tag", 
                        "value": str(meta_description),  # Ensure string
                        "type": "single_line_text_field",
                        "namespace": "global"
                    }
                ]
        except Exception as meta_error:
            logger.warning(f"Skipping metafields due to error: {meta_error}")
            # Continue without metafields if they cause issues
        
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
    """Generate SEO-optimized meta description - SINGLE LINE ONLY"""
    import re
    
    # Remove HTML tags and get first paragraph
    clean_content = re.sub('<[^<]+?>', '', content)
    
    # Remove all line breaks and extra spaces
    clean_content = re.sub(r'\s+', ' ', clean_content).strip()
    
    sentences = clean_content.split('.')[:2]  # First 2 sentences
    base_description = '. '.join(sentences).strip()
    
    # Ensure it includes key elements and stays under 160 chars
    if len(base_description) > 140:
        base_description = base_description[:140] + "..."
    
    # Add NeonXpert branding if not present
    if "NeonXpert" not in base_description:
        base_description = f"{base_description} | NeonXpert"
    
    # CRITICAL: Ensure single line and remove any problematic characters
    final_description = re.sub(r'[\r\n\t]', ' ', base_description)
    final_description = re.sub(r'\s+', ' ', final_description).strip()
    
    return final_description[:160]  # Meta description limit

def get_smart_tags(title, category):
    """Enhanced smart tags with SEO optimization"""
    tags = [category, "NeonXpert", str(datetime.datetime.now().year)]
    
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
    if any(word in title_lower for word in ["lgbtq", "pride", "rainbow"]):
        tags.append("LGBTQ Pride")
    
    # Add industry and SEO tags
    tags.extend(["LED Signs", "Business Marketing", "Storefront Design", "SEO Optimized"])
    
    return ", ".join(tags[:10])  # Limit to 10 tags for better organization

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port), '', content)  # Remove incomplete HTML tags
        content = re.sub(r'\s*```\s*

@app.route('/publish_blog', methods=['POST'])
def publish_blog():
    """Enhanced publish with better error handling and consistent content"""
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
        
        # Clean title for metafields (ensure single line)
        clean_title = re.sub(r'[\r\n\t]', ' ', topic['title'])
        clean_title = re.sub(r'\s+', ' ', clean_title).strip()[:60]
        
        # Enhanced blog data with full SEO optimization
        blog_data = {
            "article": {
                "title": topic['title'],
                "author": "NeonXpert Team",  # Professional team attribution
                "body_html": blog_html,  # Now includes Rich Schema
                "blog_id": int(BLOG_ID),
                "tags": get_smart_tags(topic['title'], topic['category']),
                "published": True,
                "handle": slug,
                "summary": meta_description,  # Optimized meta description
            }
        }
        
        # Add metafields only if we have valid values (Shopify is picky about these)
        try:
            if clean_title and meta_description:
                blog_data["article"]["metafields"] = [
                    {
                        "key": "title_tag",
                        "value": str(clean_title),  # Ensure string
                        "type": "single_line_text_field",
                        "namespace": "global"
                    },
                    {
                        "key": "description_tag", 
                        "value": str(meta_description),  # Ensure string
                        "type": "single_line_text_field",
                        "namespace": "global"
                    }
                ]
        except Exception as meta_error:
            logger.warning(f"Skipping metafields due to error: {meta_error}")
            # Continue without metafields if they cause issues
        
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
    """Generate SEO-optimized meta description - SINGLE LINE ONLY"""
    import re
    
    # Remove HTML tags and get first paragraph
    clean_content = re.sub('<[^<]+?>', '', content)
    
    # Remove all line breaks and extra spaces
    clean_content = re.sub(r'\s+', ' ', clean_content).strip()
    
    sentences = clean_content.split('.')[:2]  # First 2 sentences
    base_description = '. '.join(sentences).strip()
    
    # Ensure it includes key elements and stays under 160 chars
    if len(base_description) > 140:
        base_description = base_description[:140] + "..."
    
    # Add NeonXpert branding if not present
    if "NeonXpert" not in base_description:
        base_description = f"{base_description} | NeonXpert"
    
    # CRITICAL: Ensure single line and remove any problematic characters
    final_description = re.sub(r'[\r\n\t]', ' ', base_description)
    final_description = re.sub(r'\s+', ' ', final_description).strip()
    
    return final_description[:160]  # Meta description limit

def get_smart_tags(title, category):
    """Enhanced smart tags with SEO optimization"""
    tags = [category, "NeonXpert", str(datetime.datetime.now().year)]
    
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
    if any(word in title_lower for word in ["lgbtq", "pride", "rainbow"]):
        tags.append("LGBTQ Pride")
    
    # Add industry and SEO tags
    tags.extend(["LED Signs", "Business Marketing", "Storefront Design", "SEO Optimized"])
    
    return ", ".join(tags[:10])  # Limit to 10 tags for better organization

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port), '', content)  # Remove trailing markdown
        
        # Clean up extra whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # Reduce excessive line breaks
        content = content.strip()
        
        return content
        
    except Exception as e:
        logger.warning(f"Content cleaning failed: {e}")
        return content

@app.route('/publish_blog', methods=['POST'])
def publish_blog():
    """Enhanced publish with better error handling and consistent content"""
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
        
        # Clean title for metafields (ensure single line)
        clean_title = re.sub(r'[\r\n\t]', ' ', topic['title'])
        clean_title = re.sub(r'\s+', ' ', clean_title).strip()[:60]
        
        # Enhanced blog data with full SEO optimization
        blog_data = {
            "article": {
                "title": topic['title'],
                "author": "NeonXpert Team",  # Professional team attribution
                "body_html": blog_html,  # Now includes Rich Schema
                "blog_id": int(BLOG_ID),
                "tags": get_smart_tags(topic['title'], topic['category']),
                "published": True,
                "handle": slug,
                "summary": meta_description,  # Optimized meta description
            }
        }
        
        # Add metafields only if we have valid values (Shopify is picky about these)
        try:
            if clean_title and meta_description:
                blog_data["article"]["metafields"] = [
                    {
                        "key": "title_tag",
                        "value": str(clean_title),  # Ensure string
                        "type": "single_line_text_field",
                        "namespace": "global"
                    },
                    {
                        "key": "description_tag", 
                        "value": str(meta_description),  # Ensure string
                        "type": "single_line_text_field",
                        "namespace": "global"
                    }
                ]
        except Exception as meta_error:
            logger.warning(f"Skipping metafields due to error: {meta_error}")
            # Continue without metafields if they cause issues
        
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
    """Generate SEO-optimized meta description - SINGLE LINE ONLY"""
    import re
    
    # Remove HTML tags and get first paragraph
    clean_content = re.sub('<[^<]+?>', '', content)
    
    # Remove all line breaks and extra spaces
    clean_content = re.sub(r'\s+', ' ', clean_content).strip()
    
    sentences = clean_content.split('.')[:2]  # First 2 sentences
    base_description = '. '.join(sentences).strip()
    
    # Ensure it includes key elements and stays under 160 chars
    if len(base_description) > 140:
        base_description = base_description[:140] + "..."
    
    # Add NeonXpert branding if not present
    if "NeonXpert" not in base_description:
        base_description = f"{base_description} | NeonXpert"
    
    # CRITICAL: Ensure single line and remove any problematic characters
    final_description = re.sub(r'[\r\n\t]', ' ', base_description)
    final_description = re.sub(r'\s+', ' ', final_description).strip()
    
    return final_description[:160]  # Meta description limit

def get_smart_tags(title, category):
    """Enhanced smart tags with SEO optimization"""
    tags = [category, "NeonXpert", str(datetime.datetime.now().year)]
    
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
    if any(word in title_lower for word in ["lgbtq", "pride", "rainbow"]):
        tags.append("LGBTQ Pride")
    
    # Add industry and SEO tags
    tags.extend(["LED Signs", "Business Marketing", "Storefront Design", "SEO Optimized"])
    
    return ", ".join(tags[:10])  # Limit to 10 tags for better organization

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
