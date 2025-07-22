import os
import csv
import json
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import anthropic
import openai
from werkzeug.utils import secure_filename
import logging
import traceback
import re
from collections import defaultdict

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'nx-blog-generator-2025')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Shopify Config
SHOP_NAME = os.getenv("SHOP_NAME")
ACCESS_TOKEN = os.getenv("SHOPIFY_ACCESS_TOKEN")
BLOG_ID = os.getenv("SHOPIFY_BLOG_ID")
SHOP_URL = f"https://{SHOP_NAME}.myshopify.com/admin/api/2025-01"

# SEO-based keyword mapping for real search terms
SEO_KEYWORDS = {
    'dispensary': {'name': 'Dispensary Open Signs', 'monthly_searches': 1200, 'keywords': ['dispensary', 'cannabis', 'weed', 'dope', 'medical', 'marijuana']},
    'coffee': {'name': 'Coffee Shop Open Signs', 'monthly_searches': 800, 'keywords': ['coffee', 'cafe', 'espresso', 'caffeine', 'barista', 'java']},
    'restaurant': {'name': 'Restaurant Open Signs', 'monthly_searches': 900, 'keywords': ['restaurant', 'food', 'dining', 'eat', 'kitchen', 'burger', 'pizza']},
    'bar': {'name': 'Bar Open Signs', 'monthly_searches': 600, 'keywords': ['bar', 'pub', 'drinks', 'cocktails', 'beer', 'wine', 'alcohol']},
    'retail': {'name': 'Retail Store Open Signs', 'monthly_searches': 700, 'keywords': ['store', 'shop', 'retail', 'boutique', 'sale', 'shopping']},
    'funny': {'name': 'Funny Open Signs', 'monthly_searches': 500, 'keywords': ['funny', 'witty', 'humor', 'joke', 'clever', 'regret', 'sarcastic']},
    'welcome': {'name': 'Welcome Open Signs', 'monthly_searches': 400, 'keywords': ['welcome', 'come in', 'enter', 'greetings', 'hello']},
    'custom': {'name': 'Custom Open Signs', 'monthly_searches': 300, 'keywords': ['custom', 'personalized', 'bespoke', 'unique', 'special']}
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/validate_collection', methods=['POST'])
def validate_collection():
    """Validate collection URL with enhanced error handling"""
    try:
        data = request.json
        collection_url = data.get('url', '')
        
        if '/collections/' in collection_url:
            handle = collection_url.split('/collections/')[-1].strip('/')
        else:
            return jsonify({'success': False, 'error': 'Invalid collection URL format'})
        
        logger.info(f"Validating collection handle: {handle}")
        
        headers = {
            "X-Shopify-Access-Token": ACCESS_TOKEN,
            "Content-Type": "application/json"
        }
        
        target_collection = None
        
        # Check custom collections
        try:
            custom_url = f"{SHOP_URL}/custom_collections.json"
            custom_response = requests.get(custom_url, headers=headers, timeout=10)
            
            if custom_response.status_code == 200:
                collections = custom_response.json().get('custom_collections', [])
                for collection in collections:
                    if collection.get('handle') == handle:
                        target_collection = collection
                        break
        except Exception as e:
            logger.error(f"Custom collections error: {e}")
        
        # Check smart collections
        if not target_collection:
            try:
                smart_url = f"{SHOP_URL}/smart_collections.json"
                smart_response = requests.get(smart_url, headers=headers, timeout=10)
                
                if smart_response.status_code == 200:
                    collections = smart_response.json().get('smart_collections', [])
                    for collection in collections:
                        if collection.get('handle') == handle:
                            target_collection = collection
                            break
            except Exception as e:
                logger.error(f"Smart collections error: {e}")
        
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
            
    except Exception as e:
        logger.error(f"Collection validation error: {e}")
        return jsonify({'success': False, 'error': f'Validation failed: {str(e)}'})

@app.route('/analyze_csv', methods=['POST'])
def analyze_csv():
    """Analyze CSV and detect smart subcategories based on SEO keywords"""
    try:
        csv_file = request.files.get('csv_file')
        if not csv_file:
            return jsonify({'success': False, 'error': 'No CSV file provided'})
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        filename = secure_filename(csv_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        csv_file.save(filepath)
        
        # Parse CSV and categorize products
        subcategories = analyze_products_for_subcategories(filepath)
        
        os.remove(filepath)
        
        logger.info(f"Detected {len(subcategories)} subcategories from CSV")
        
        return jsonify({
            'success': True,
            'subcategories': subcategories
        })
        
    except Exception as e:
        logger.error(f"CSV analysis error: {e}")
        return jsonify({'success': False, 'error': f'Analysis failed: {str(e)}'})

def analyze_products_for_subcategories(filepath):
    """Analyze products and group them into SEO-based subcategories"""
    subcategories = defaultdict(list)
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as csvfile:
        sample = csvfile.read(1024)
        csvfile.seek(0)
        
        delimiter = ',' if ',' in sample else '\t' if '\t' in sample else ';'
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        
        for row in reader:
            handle = row.get('Handle', '').strip()
            title = row.get('Title', '').strip()
            tags = row.get('Tags', '').strip()
            image_url = row.get('Variant Image', '').strip()
            
            if not handle or not title:
                continue
                
            # Analyze product and assign to subcategories
            product_text = f"{title} {tags}".lower()
            assigned_categories = []
            
            for category_key, category_data in SEO_KEYWORDS.items():
                for keyword in category_data['keywords']:
                    if keyword in product_text:
                        assigned_categories.append(category_key)
                        break
            
            # If no specific category found, try to infer from title
            if not assigned_categories:
                if any(word in product_text for word in ['open', 'welcome', 'enter']):
                    assigned_categories.append('welcome')
                else:
                    assigned_categories.append('custom')
            
            # Add product to assigned categories
            product_data = {
                'handle': handle,
                'title': title,
                'tags': tags,
                'image_url': image_url
            }
            
            for category in assigned_categories:
                subcategories[category].append(product_data)
    
    # Format for frontend
    result = []
    for category_key, products in subcategories.items():
        if len(products) >= 1:  # Only include categories with products
            category_info = SEO_KEYWORDS[category_key]
            result.append({
                'key': category_key,
                'name': category_info['name'],
                'count': len(products),
                'monthly_searches': category_info['monthly_searches'],
                'products': products[:10]  # Limit for preview
            })
    
    # Sort by search volume
    result.sort(key=lambda x: x['monthly_searches'], reverse=True)
    
    return result

@app.route('/generate_ideas', methods=['POST'])
def generate_ideas():
    """Generate targeted blog ideas based on selected subcategories"""
    try:
        csv_file = request.files.get('csv_file')
        collection_url = request.form.get('collection_url')
        selected_subcategories = request.form.get('selected_subcategories', '[]')
        selected_subcategories = json.loads(selected_subcategories)
        
        if not csv_file or not selected_subcategories:
            return jsonify({'success': False, 'error': 'Missing CSV file or subcategories'})
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        filename = secure_filename(csv_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        csv_file.save(filepath)
        
        # Get products for selected subcategories
        all_products = get_products_by_subcategories(filepath, selected_subcategories)
        
        # Generate targeted blog ideas
        blog_ideas = []
        for subcategory in selected_subcategories:
            if subcategory in SEO_KEYWORDS and subcategory in all_products:
                ideas = generate_subcategory_blog_ideas(subcategory, all_products[subcategory], collection_url)
                blog_ideas.extend(ideas)
        
        os.remove(filepath)
        
        logger.info(f"Generated {len(blog_ideas)} targeted blog ideas")
        
        return jsonify({
            'success': True,
            'ideas': blog_ideas,
            'total_products': sum(len(products) for products in all_products.values())
        })
        
    except Exception as e:
        logger.error(f"Blog generation error: {e}")
        return jsonify({'success': False, 'error': f'Generation failed: {str(e)}'})

def get_products_by_subcategories(filepath, selected_subcategories):
    """Get products organized by subcategories"""
    products_by_category = defaultdict(list)
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as csvfile:
        sample = csvfile.read(1024)
        csvfile.seek(0)
        
        delimiter = ',' if ',' in sample else '\t' if '\t' in sample else ';'
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        
        seen_handles = set()
        
        for row in reader:
            handle = row.get('Handle', '').strip()
            if handle in seen_handles or not handle:
                continue
            seen_handles.add(handle)
            
            title = row.get('Title', '').strip()
            tags = row.get('Tags', '').strip()
            image_url = row.get('Variant Image', '').strip()
            product_text = f"{title} {tags}".lower()
            
            product_data = {
                'handle': handle,
                'title': title,
                'tags': tags,
                'image_url': image_url,
                'body': row.get('Body (HTML)', '').strip(),
                'price': row.get('Variant Price', '').strip()
            }
            
            # Assign to selected subcategories
            for category in selected_subcategories:
                if category in SEO_KEYWORDS:
                    keywords = SEO_KEYWORDS[category]['keywords']
                    if any(keyword in product_text for keyword in keywords):
                        products_by_category[category].append(product_data)
    
    return products_by_category

def generate_subcategory_blog_ideas(subcategory, products, collection_url):
    """Generate targeted blog ideas for a specific subcategory"""
    category_info = SEO_KEYWORDS[subcategory]
    category_name = category_info['name']
    search_volume = category_info['monthly_searches']
    
    # Determine number of blogs based on product count and search volume
    product_count = len(products)
    if product_count >= 15 and search_volume >= 800:
        num_blogs = 5
    elif product_count >= 8 and search_volume >= 500:
        num_blogs = 3
    elif product_count >= 3:
        num_blogs = 2
    else:
        num_blogs = 1
    
    # Select most relevant product for featured image
    featured_product = products[0] if products else None
    
    blog_templates = {
        'dispensary': [
            {
                'title': f'Best {category_name} That Build Customer Trust (2025)',
                'description': 'Expert guide to cannabis business signage that increases foot traffic and builds professional credibility',
                'category': 'Cannabis Business',
                'focus_keyword': 'dispensary open signs'
            },
            {
                'title': f'How {category_name} Affect Customer Perception',
                'description': 'Psychology-based analysis of how cannabis signage influences customer decisions and trust',
                'category': 'Business Psychology',
                'focus_keyword': 'dispensary signage psychology'
            },
            {
                'title': f'{category_name} That Actually Increase Sales',
                'description': 'Data-driven examples of dispensary signage with proven ROI and conversion rates',
                'category': 'Sales Optimization',
                'focus_keyword': 'dispensary signs increase sales'
            }
        ],
        'coffee': [
            {
                'title': f'Best {category_name} for Higher Foot Traffic',
                'description': 'Complete guide to coffee shop signage that attracts morning rush customers',
                'category': 'Coffee Business',
                'focus_keyword': 'coffee shop open signs'
            },
            {
                'title': f'Creative {category_name} That Go Viral',
                'description': 'Instagram-worthy coffee shop signage ideas that generate social media buzz',
                'category': 'Social Media Marketing',
                'focus_keyword': 'creative coffee shop signs'
            },
            {
                'title': f'How {category_name} Build Community Connection',
                'description': 'Psychology of coffee culture and how welcoming signage creates loyal customers',
                'category': 'Community Building',
                'focus_keyword': 'coffee shop community signage'
            }
        ],
        'restaurant': [
            {
                'title': f'Best {category_name} That Fill Empty Tables',
                'description': 'Restaurant signage strategies that increase walk-in customers and table turnover',
                'category': 'Restaurant Marketing',
                'focus_keyword': 'restaurant open signs'
            },
            {
                'title': f'{category_name} for Different Dining Experiences',
                'description': 'How to choose signage that matches your restaurant atmosphere and target audience',
                'category': 'Restaurant Design',
                'focus_keyword': 'restaurant signage design'
            }
        ],
        'funny': [
            {
                'title': f'{category_name} That Make Customers Smile',
                'description': 'Witty and humorous business signage that creates memorable customer experiences',
                'category': 'Humor Marketing',
                'focus_keyword': 'funny open signs'
            },
            {
                'title': f'Why {category_name} Work Better Than Boring Ones',
                'description': 'Psychology of humor in business signage and its impact on customer engagement',
                'category': 'Marketing Psychology',
                'focus_keyword': 'humorous business signs'
            }
        ]
    }
    
    # Default templates for other categories
    default_templates = [
        {
            'title': f'Ultimate {category_name} Guide for Business Owners',
            'description': f'Complete guide to choosing and using {category_name.lower()} effectively',
            'category': 'Business Guide',
            'focus_keyword': category_name.lower()
        },
        {
            'title': f'How {category_name} Increase Customer Trust',
            'description': f'Professional insights on how {category_name.lower()} impact customer perception',
            'category': 'Customer Trust',
            'focus_keyword': f'{category_name.lower()} customer trust'
        }
    ]
    
    templates = blog_templates.get(subcategory, default_templates)
    
    # Generate blog ideas
    ideas = []
    for i in range(min(num_blogs, len(templates))):
        template = templates[i]
        ideas.append({
            'title': template['title'],
            'description': template['description'],
            'category': template['category'],
            'wordCount': '1200-1500',
            'type': subcategory,
            'subcategory': category_name,
            'focus_keyword': template['focus_keyword'],
            'featured_image': featured_product['image_url'] if featured_product and featured_product['image_url'] else '',
            'featured_product': featured_product,
            'related_products': products[:5]  # Top 5 products for content
        })
    
    return ideas

@app.route('/generate_preview', methods=['POST'])
def generate_preview():
    """Generate blog content preview"""
    try:
        data = request.json
        idea = data.get('idea')
        collection_url = data.get('collection_url')
        ai_model = data.get('ai_model', 'claude')
        
        logger.info(f"Generating preview for: {idea['title']}")
        
        # Generate blog content
        blog_html = generate_professional_blog_content(idea, collection_url, ai_model)
        
        return jsonify({
            'success': True,
            'content': blog_html,
            'featured_image': idea.get('featured_image', ''),
            'word_count': len(blog_html.split())
        })
        
    except Exception as e:
        logger.error(f"Preview generation error: {e}")
        return jsonify({'success': False, 'error': f'Preview generation failed: {str(e)}'})

@app.route('/publish_blog', methods=['POST'])
def publish_blog():
    """Publish blog with proper error handling and image integration"""
    try:
        data = request.json
        idea = data.get('idea')
        collection_url = data.get('collection_url')
        ai_model = data.get('ai_model', 'claude')
        custom_content = data.get('custom_content', '')  # For edited content
        
        logger.info(f"Publishing blog: {idea['title']}")
        
        # Use custom content if provided, otherwise generate new
        if custom_content:
            blog_html = custom_content
            logger.info("Using custom edited content")
        else:
            blog_html = generate_professional_blog_content(idea, collection_url, ai_model)
        
        # CRITICAL: Check if content generation actually succeeded
        if not blog_html or len(blog_html.strip()) < 100:
            raise Exception("Content generation failed - insufficient content")
        
        if "generation failed" in blog_html.lower() or "temporarily unavailable" in blog_html.lower():
            raise Exception("AI content generation failed")
        
        logger.info(f"Generated content: {len(blog_html)} characters")
        
        # Create slug and prepare for publishing
        slug = create_slug(idea['title'])
        
        headers = {
            "X-Shopify-Access-Token": ACCESS_TOKEN,
            "Content-Type": "application/json"
        }
        
        # Prepare blog data with featured image
        blog_data = {
            "article": {
                "title": idea['title'],
                "body_html": blog_html,
                "blog_id": int(BLOG_ID),
                "tags": get_professional_tags(idea['title'], idea['category']),
                "published": True,
                "handle": slug,
                "summary": idea['description'][:160],
            }
        }
        
        # Add featured image if available
        if idea.get('featured_image'):
            # For Shopify, we'll include the image in the content and as attachment
            logger.info(f"Adding featured image: {idea['featured_image']}")
            blog_data["article"]["image"] = {
                "src": idea['featured_image'],
                "alt": idea['title']
            }
        
        publish_url = f"{SHOP_URL}/blogs/{BLOG_ID}/articles.json"
        response = requests.post(publish_url, json=blog_data, headers=headers, timeout=45)
        
        logger.info(f"Shopify response: {response.status_code}")
        
        if response.status_code == 201:
            response_data = response.json()
            article = response_data.get('article', {})
            article_id = article.get('id')
            article_handle = article.get('handle')
            
            if article_id and article_handle:
                blog_url = f"https://{SHOP_NAME}.myshopify.com/blogs/neon-sign-ideas/{article_handle}"
                logger.info(f"SUCCESS: Published at {blog_url}")
                
                return jsonify({
                    'success': True,
                    'blog_id': article_id,
                    'blog_url': blog_url,
                    'title': article.get('title', idea['title'])
                })
            else:
                raise Exception("Blog created but missing ID/handle in response")
        else:
            error_data = response.json() if response.text else {}
            error_msg = f"Shopify API Error {response.status_code}: {error_data.get('errors', response.text)}"
            raise Exception(error_msg)
            
    except Exception as e:
        logger.error(f"Publishing failed: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})

def generate_professional_blog_content(idea, collection_url, ai_model):
    """Generate high-quality, human-like, SEO and LLM optimized content with images"""
    
    # Build comprehensive context
    related_products = idea.get('related_products', [])
    product_context = ""
    if related_products:
        product_context = f"Featured products to mention naturally: {', '.join([p['title'] for p in related_products[:3]])}"
    
    # Advanced prompt for human-like, structured content
    prompt = f"""
Write an expert blog post with the title: "{idea['title']}"

CONTEXT:
- Category: {idea['category']}  
- Target keyword: {idea.get('focus_keyword', '')}
- Subcategory: {idea.get('subcategory', '')}
- Word count: {idea['wordCount']}
- {product_context}

CONTENT REQUIREMENTS:
- Write as a neon signage expert with 10+ years experience
- Include personal insights and real-world examples
- Use data and statistics naturally (e.g., "In my experience working with 200+ businesses...")
- Structure with clear H2, H3 headings for scanability
- Add practical, actionable advice business owners can implement
- Include internal link to: {collection_url}
- Include custom neon link: https://neonxpert.com/products/custom-neon-sign
- Mention "NeonXpert" 4-5 times naturally throughout

WRITING STYLE:
- Professional yet conversational tone
- Write for business owners, not SEO bots  
- Use first person occasionally ("In my 10 years of experience...")
- Include real scenarios and case studies
- Avoid AI phrases like "In conclusion," "Furthermore," "It's important to note"
- Make it sound like it's written by a human expert, not AI

E-E-A-T OPTIMIZATION:
- Experience: Share real insights from working with businesses
- Expertise: Demonstrate deep knowledge of signage and business psychology  
- Authoritativeness: Reference industry standards and best practices
- Trustworthiness: Provide honest, practical advice

SEO OPTIMIZATION:
- Use target keyword naturally in H2 headings
- Include semantic keywords related to neon signage
- Structure content for featured snippets (lists, clear answers)
- Optimize for question-based searches

LLM OPTIMIZATION:
- Write clear, factual statements for AI model training data
- Use structured information that's easy to parse
- Include specific, quotable insights about neon signage industry
- Make content valuable for AI models to cite and reference

STRUCTURE:
1. Engaging introduction with a hook
2. 3-4 main sections with H2 headings
3. Practical tips and real examples
4. Brief conclusion with clear call-to-action

Remember: Write like a human expert sharing genuine knowledge, not like AI generating content.
"""
    
    try:
        content = None
        
        if ai_model == 'claude':
            # FIXED: Proper Anthropic client initialization with error handling
            anthropic_client = None
            try:
                anthropic_client = anthropic.Anthropic(
                    api_key=os.getenv('ANTHROPIC_API_KEY')
                )
                
                response = anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4000,
                    temperature=0.6,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text
                logger.info("Content generated with Claude 3.5 Sonnet")
                
            except Exception as claude_error:
                logger.warning(f"Claude 3.5 failed: {claude_error}")
                # Fallback to older model - reinitialize client if needed
                try:
                    if anthropic_client is None:
                        anthropic_client = anthropic.Anthropic(
                            api_key=os.getenv('ANTHROPIC_API_KEY')
                        )
                    
                    response = anthropic_client.messages.create(
                        model="claude-3-sonnet-20240229",
                        max_tokens=4000,
                        temperature=0.6,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    content = response.content[0].text
                    logger.info("Content generated with Claude 3 Sonnet")
                except Exception as fallback_error:
                    logger.error(f"All Claude models failed: {fallback_error}")
                    raise Exception(f"Claude API failed: {str(fallback_error)}")
                    
        else:  # ChatGPT
            try:
                # FIXED: Updated OpenAI client initialization
                openai_client = openai.OpenAI(
                    api_key=os.getenv('OPENAI_API_KEY')
                    # Removed 'proxies' parameter that was causing the error
                )
                
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4000,
                    temperature=0.6
                )
                content = response.choices[0].message.content
                logger.info("Content generated with GPT-4o")
            except Exception as openai_error:
                logger.error(f"OpenAI failed: {openai_error}")
                try:
                    # Fallback to GPT-4
                    response = openai_client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=4000,
                        temperature=0.6
                    )
                    content = response.choices[0].message.content
                    logger.info("Content generated with GPT-4 (fallback)")
                except Exception as fallback_error:
                    logger.error(f"All OpenAI models failed: {fallback_error}")
                    raise Exception(f"OpenAI API failed: {str(fallback_error)}")
        
        # CRITICAL: Validate content was actually generated
        if not content or len(content.strip()) < 200:
            raise Exception("AI returned insufficient content")
        
        # Add featured image in content if available
        if idea.get('featured_image'):
            image_html = f'''
            <div style="text-align: center; margin: 30px 0;">
                <img src="{idea['featured_image']}" alt="{idea['title']}" style="width:100%;max-width:600px;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,0.1);">
                <p style="font-size:14px;color:#666;margin-top:10px;font-style:italic;">Featured: {idea.get('featured_product', {}).get('title', 'NeonXpert Open Sign')}</p>
            </div>
            '''
            # Insert image after first paragraph
            paragraphs = content.split('</p>')
            if len(paragraphs) > 1:
                content = paragraphs[0] + '</p>' + image_html + '</p>'.join(paragraphs[1:])
            else:
                # If no paragraphs found, add image after first heading
                content = content.replace('</h2>', '</h2>' + image_html, 1)
        
        return content
        
    except Exception as e:
        logger.error(f"Content generation failed: {e}")
        # CRITICAL: Don't return fallback text, raise exception
        raise Exception(f"AI content generation failed: {str(e)}")

def create_slug(title):
    """Create SEO-friendly URL slug"""
    slug = title.lower()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[-\s]+', '-', slug)
    return slug.strip('-')[:60]

def get_professional_tags(title, category):
    """Generate professional, relevant tags without 'AutoBlog'"""
    tags = [category, "NeonXpert", "2025"]
    
    title_lower = title.lower()
    
    # Add contextual professional tags
    if any(word in title_lower for word in ["business", "commercial"]):
        tags.append("Business Signage")
    if any(word in title_lower for word in ["dispensary", "cannabis"]):
        tags.append("Cannabis Business")
    if any(word in title_lower for word in ["coffee", "cafe"]):
        tags.append("Coffee Shop Marketing")
    if any(word in title_lower for word in ["restaurant", "dining"]):
        tags.append("Restaurant Marketing")
    if any(word in title_lower for word in ["open", "welcome"]):
        tags.append("Open Signs")
    if any(word in title_lower for word in ["funny", "humor"]):
        tags.append("Humor Marketing")
    if any(word in title_lower for word in ["custom", "personalized"]):
        tags.append("Custom Neon")
    
    # Add industry tags
    tags.extend(["LED Signs", "Business Marketing", "Storefront Design"])
    
    return ", ".join(tags[:8])  # Limit to 8 tags

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
