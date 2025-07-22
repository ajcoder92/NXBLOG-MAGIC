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

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'nx-blog-generator-2025')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Setup logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Clients
claude_client = None
openai_client = None

# MODERN SHOPIFY CONFIG
SHOP_NAME = os.getenv("SHOP_NAME")
ACCESS_TOKEN = os.getenv("SHOPIFY_ACCESS_TOKEN")
BLOG_ID = os.getenv("SHOPIFY_BLOG_ID")
SHOP_URL = f"https://{SHOP_NAME}.myshopify.com/admin/api/2025-01"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/validate_collection', methods=['POST'])
def validate_collection():
    """Validate collection URL using modern Shopify API with detailed logging"""
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
            logger.info(f"Requesting: {custom_url}")
            custom_response = requests.get(custom_url, headers=headers, timeout=10)
            logger.info(f"Custom collections response: {custom_response.status_code}")
            
            if custom_response.status_code == 200:
                collections = custom_response.json().get('custom_collections', [])
                logger.info(f"Found {len(collections)} custom collections")
                for collection in collections:
                    if collection.get('handle') == handle:
                        target_collection = collection
                        logger.info(f"Found collection: {collection.get('title')}")
                        break
            else:
                logger.error(f"Custom collections error: {custom_response.text}")
        except Exception as e:
            logger.error(f"Custom collections exception: {e}")
        
        # Check smart collections if not found
        if not target_collection:
            try:
                smart_url = f"{SHOP_URL}/smart_collections.json"
                logger.info(f"Requesting: {smart_url}")
                smart_response = requests.get(smart_url, headers=headers, timeout=10)
                logger.info(f"Smart collections response: {smart_response.status_code}")
                
                if smart_response.status_code == 200:
                    collections = smart_response.json().get('smart_collections', [])
                    logger.info(f"Found {len(collections)} smart collections")
                    for collection in collections:
                        if collection.get('handle') == handle:
                            target_collection = collection
                            logger.info(f"Found collection: {collection.get('title')}")
                            break
                else:
                    logger.error(f"Smart collections error: {smart_response.text}")
            except Exception as e:
                logger.error(f"Smart collections exception: {e}")
        
        if target_collection:
            description = target_collection.get('body_html', '')
            if description:
                import re
                description = re.sub('<[^<]+?>', '', description)
                description = description.strip()[:200] + '...' if len(description) > 200 else description
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
                'error': f'Collection "{handle}" not found. Please check the URL.'
            })
            
    except Exception as e:
        logger.error(f"Collection validation error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': f'Validation error: {str(e)}'})

@app.route('/generate_ideas', methods=['POST'])
def generate_ideas():
    """Generate blog ideas with better variety"""
    try:
        csv_file = request.files.get('csv_file')
        collection_url = request.form.get('collection_url')
        ai_model = request.form.get('ai_model')
        collection_type = request.form.get('collection_type')
        custom_collection_type = request.form.get('custom_collection_type', '')
        
        # Use custom type if provided
        if collection_type == 'custom' and custom_collection_type.strip():
            collection_type = custom_collection_type.strip()
        
        if not csv_file:
            return jsonify({'success': False, 'error': 'No CSV file uploaded'})
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        filename = secure_filename(csv_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        csv_file.save(filepath)
        
        # Process CSV
        unique_products = []
        seen_handles = set()
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as csvfile:
            sample = csvfile.read(1024)
            csvfile.seek(0)
            
            delimiter = ','
            if '\t' in sample:
                delimiter = '\t'
            elif ';' in sample:
                delimiter = ';'
            
            reader = csv.DictReader(csvfile, delimiter=delimiter)
            
            for row in reader:
                handle = row.get('Handle', '').strip()
                if handle and handle not in seen_handles:
                    seen_handles.add(handle)
                    unique_products.append({
                        'handle': handle,
                        'title': row.get('Title', '').strip(),
                        'body': row.get('Body (HTML)', '').strip(),
                        'tags': row.get('Tags', '').strip(),
                        'image_url': row.get('Variant Image', '').strip(),
                        'price': row.get('Variant Price', '').strip()
                    })
        
        # Generate blog ideas
        blog_ideas = generate_blog_ideas_for_type(collection_type, unique_products, collection_url)
        
        os.remove(filepath)
        
        logger.info(f"Generated {len(blog_ideas)} blog ideas for {len(unique_products)} products")
        
        return jsonify({
            'success': True,
            'ideas': blog_ideas,
            'product_count': len(unique_products)
        })
        
    except Exception as e:
        logger.error(f"Blog generation error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': f'Generation error: {str(e)}'})

def generate_blog_ideas_for_type(collection_type, products, collection_url):
    """Generate diverse blog ideas"""
    
    collection_name = collection_url.split('/collections/')[-1].replace('-', ' ').title()
    
    if collection_type == 'business' or 'open' in collection_name.lower():
        return [
            {
                'title': f'Best {collection_name} for Small Businesses That Increase Foot Traffic (2025)',
                'description': 'Comprehensive guide to choosing business signage that drives customer engagement',
                'category': 'Business Marketing',
                'wordCount': '1200-1500',
                'type': 'marketing'
            },
            {
                'title': f'How to Choose {collection_name} for Your Business (Complete Guide)',
                'description': 'Expert buyer guide with practical tips for selecting perfect business signage',
                'category': 'How-To Guide',
                'wordCount': '1400-1700',
                'type': 'howto'
            },
            {
                'title': f'10 {collection_name} Ideas That Actually Boost Sales',
                'description': 'Data-driven examples of effective business signage with real results',
                'category': 'Listicle',
                'wordCount': '1000-1300',
                'type': 'listicle'
            },
            {
                'title': f'{collection_name} Psychology: How Customers Decide to Enter Your Store',
                'description': 'Science-backed insights into customer decision-making and storefront psychology',
                'category': 'Customer Psychology',
                'wordCount': '1300-1600',
                'type': 'psychology'
            },
            {
                'title': f'Instagram-Worthy {collection_name} That Go Viral in 2025',
                'description': 'Social media optimization guide for business signage that generates shares',
                'category': 'Social Media Marketing',
                'wordCount': '900-1200',
                'type': 'social'
            },
            {
                'title': f'Why Generic {collection_name} Don\'t Work (And What Does)',
                'description': 'Common signage mistakes that kill business and how to avoid them',
                'category': 'Problem-Solving',
                'wordCount': '1100-1400',
                'type': 'problem'
            },
            {
                'title': f'{collection_name} Trends Dominating 2025: What Businesses Need to Know',
                'description': 'Latest signage trends and future predictions for business marketing',
                'category': 'Industry Trends',
                'wordCount': '1000-1300',
                'type': 'trends'
            }
        ]
    
    elif collection_type == 'wedding':
        return [
            {
                'title': f'30 Stunning {collection_name} Ideas That Will Make Guests Swoon (2025)',
                'description': 'Romantic wedding signage inspiration with real examples',
                'category': 'Wedding Inspiration',
                'wordCount': '1500-1800',
                'type': 'inspiration'
            },
            {
                'title': f'How to Choose {collection_name} That Match Your Wedding Theme',
                'description': 'Wedding planning guide for selecting perfect signage',
                'category': 'Wedding Planning',
                'wordCount': '1200-1500',
                'type': 'planning'
            },
            {
                'title': f'Budget-Friendly {collection_name} That Look Expensive',
                'description': 'Affordable wedding signage options that maintain premium appearance',
                'category': 'Wedding Budget',
                'wordCount': '1000-1300',
                'type': 'budget'
            }
        ]
    
    else:
        # Custom collection type
        return [
            {
                'title': f'Ultimate {collection_name} Guide: Everything You Need to Know (2025)',
                'description': f'Comprehensive guide covering everything about {collection_name}',
                'category': 'Complete Guide',
                'wordCount': '1400-1700',
                'type': 'guide'
            },
            {
                'title': f'How to Choose the Perfect {collection_name} for Your Needs',
                'description': 'Expert selection guide with practical tips',
                'category': 'Buying Guide',
                'wordCount': '1200-1500',
                'type': 'howto'
            },
            {
                'title': f'10 Creative {collection_name} Ideas That Will Inspire You',
                'description': 'Innovative ideas and creative implementations',
                'category': 'Creative Ideas',
                'wordCount': '1000-1300',
                'type': 'inspiration'
            }
        ]

@app.route('/regenerate_selected_titles', methods=['POST'])
def regenerate_selected_titles():
    """Regenerate titles for selected blogs"""
    try:
        data = request.json
        selected_indices = data.get('selected_indices', [])
        collection_url = data.get('collection_url', '')
        collection_type = data.get('collection_type', 'business')
        
        if not selected_indices:
            return jsonify({'success': False, 'error': 'No blogs selected'})
        
        logger.info(f"Regenerating titles for {len(selected_indices)} blogs")
        
        # Generate new ideas
        blog_ideas = generate_blog_ideas_for_type(collection_type, [], collection_url)
        
        # Return new titles for selected indices
        new_titles = {}
        for i, index in enumerate(selected_indices):
            if i < len(blog_ideas):
                new_titles[str(index)] = {
                    'title': blog_ideas[i]['title'],
                    'description': blog_ideas[i]['description']
                }
        
        return jsonify({
            'success': True,
            'new_titles': new_titles
        })
        
    except Exception as e:
        logger.error(f"Title regeneration error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/publish_blog', methods=['POST'])
def publish_blog():
    """Publish blog with detailed error tracking"""
    try:
        data = request.json
        idea = data.get('idea')
        collection_url = data.get('collection_url')
        ai_model = data.get('ai_model', 'claude')
        
        logger.info(f"Starting publication: {idea['title']}")
        
        # Generate blog content
        blog_html = generate_blog_content(idea, collection_url, ai_model)
        
        if not blog_html or 'generation failed' in blog_html.lower():
            logger.error("Content generation failed")
            return jsonify({'success': False, 'error': 'Failed to generate content'})
        
        logger.info(f"Generated content: {len(blog_html)} characters")
        
        # Create slug
        slug = create_slug(idea['title'])
        logger.info(f"Created slug: {slug}")
        
        # Prepare Shopify API request
        headers = {
            "X-Shopify-Access-Token": ACCESS_TOKEN,
            "Content-Type": "application/json"
        }
        
        blog_data = {
            "article": {
                "title": idea['title'],
                "body_html": blog_html,
                "blog_id": int(BLOG_ID),
                "tags": get_smart_tags(idea['title'], idea['category']),
                "published": True,
                "handle": slug,
                "summary": idea['description'][:150] + "..."
            }
        }
        
        logger.info(f"Blog data prepared: {blog_data['article']['title']}")
        
        # Make API call
        publish_url = f"{SHOP_URL}/blogs/{BLOG_ID}/articles.json"
        logger.info(f"Publishing to: {publish_url}")
        
        response = requests.post(publish_url, json=blog_data, headers=headers, timeout=30)
        
        logger.info(f"Shopify response status: {response.status_code}")
        logger.info(f"Shopify response: {response.text}")
        
        if response.status_code == 201:
            response_data = response.json()
            article = response_data.get('article', {})
            article_id = article.get('id')
            article_handle = article.get('handle')
            
            if article_id and article_handle:
                blog_url = f"https://{SHOP_NAME}.myshopify.com/blogs/neon-sign-ideas/{article_handle}"
                logger.info(f"SUCCESS: Blog published at {blog_url}")
                
                return jsonify({
                    'success': True,
                    'blog_id': article_id,
                    'blog_url': blog_url,
                    'title': article.get('title', idea['title'])
                })
            else:
                logger.error(f"Missing data in response: {response_data}")
                return jsonify({'success': False, 'error': 'Blog created but missing ID/handle'})
        else:
            error_msg = f"Shopify API Error {response.status_code}"
            try:
                error_data = response.json()
                if 'errors' in error_data:
                    error_msg += f": {error_data['errors']}"
                    logger.error(f"Shopify errors: {error_data['errors']}")
            except:
                error_msg += f": {response.text[:200]}"
            
            return jsonify({'success': False, 'error': error_msg})
            
    except Exception as e:
        logger.error(f"Publishing exception: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': f'Publishing error: {str(e)}'})

def generate_blog_content(idea, collection_url, ai_model):
    """Generate natural blog content"""
    
    prompt = f"""
    Write a comprehensive, engaging blog post with the title: "{idea['title']}"
    
    REQUIREMENTS:
    - Category: {idea['category']}
    - Target word count: {idea['wordCount']}
    - Write for NeonXpert customers interested in quality neon signs
    - Include internal link to: {collection_url}
    - Include custom neon link: https://neonxpert.com/products/custom-neon-sign
    - Use proper HTML structure with H2, H3 headings and paragraphs
    - Write in professional, engaging tone that demonstrates expertise
    - Include practical tips and actionable advice
    - Mention "NeonXpert" 4-5 times naturally throughout
    - End with compelling conclusion and call-to-action
    
    CONTENT GUIDELINES:
    - Focus on Experience, Expertise, Authority, Trust
    - Include industry insights and real examples
    - Write for humans, not search engines
    - Keep tone conversational yet professional
    - Only add FAQ if it genuinely adds value
    
    Write as a neon signage expert with deep industry knowledge.
    Make it authentic and helpful, not AI-generated sounding.
    """
    
    try:
        if ai_model == 'claude':
            global claude_client
            if claude_client is None:
                claude_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            
            models = ["claude-3-5-sonnet-20241022", "claude-3-sonnet-20240229"]
            
            for model in models:
                try:
                    response = claude_client.messages.create(
                        model=model,
                        max_tokens=4000,
                        temperature=0.7,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    content = response.content[0].text
                    logger.info(f"Content generated with Claude {model}")
                    return content
                except Exception as e:
                    logger.warning(f"Claude {model} failed: {e}")
                    continue
                    
        else:  # ChatGPT
            global openai_client
            if openai_client is None:
                openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
                temperature=0.7
            )
            content = response.choices[0].message.content
            logger.info("Content generated with GPT-4o")
            return content
            
    except Exception as e:
        logger.error(f"AI generation error: {e}")
        
    return f"<h2>Content generation temporarily unavailable</h2><p>Please try again.</p>"

def create_slug(title):
    """Create SEO-friendly URL slug"""
    import re
    slug = title.lower()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[-\s]+', '-', slug)
    return slug.strip('-')[:50]

def get_smart_tags(title, category):
    """Generate relevant tags"""
    tags = ["AutoBlog", category, "NeonXpert", "2025"]
    
    title_lower = title.lower()
    
    if any(word in title_lower for word in ["business", "commercial", "store"]):
        tags.append("Business Signs")
    if any(word in title_lower for word in ["wedding", "marriage", "bride"]):
        tags.append("Wedding Decor")
    if any(word in title_lower for word in ["home", "room", "decor"]):
        tags.append("Home Decor")
    if any(word in title_lower for word in ["open", "entrance", "welcome"]):
        tags.append("Open Signs")
    
    return ", ".join(tags)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
