import os
import csv
import json
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import anthropic
import openai
from werkzeug.utils import secure_filename
import tempfile

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'nx-blog-generator-2025')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max

# API Clients - Initialize lazily to avoid startup errors
claude_client = None
openai_client = None

# Shopify Config
SHOP_NAME = os.getenv("SHOP_NAME")
API_KEY = os.getenv("SHOPIFY_API_KEY")
PASSWORD = os.getenv("SHOPIFY_PASSWORD")
BLOG_ID = os.getenv("SHOPIFY_BLOG_ID")
SHOP_URL = f"https://{API_KEY}:{PASSWORD}@{SHOP_NAME}.myshopify.com/admin/api/2023-07"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/validate_collection', methods=['POST'])
def validate_collection():
    """Validate collection URL and extract title/description"""
    try:
        data = request.json
        collection_url = data.get('url', '')
        
        # Extract collection handle from URL
        if '/collections/' in collection_url:
            handle = collection_url.split('/collections/')[-1].strip('/')
        else:
            return jsonify({'success': False, 'error': 'Invalid collection URL'})
        
        # Fetch collection data from Shopify
        shopify_url = f"https://{SHOP_NAME}.myshopify.com/admin/api/2023-07/collections.json"
        response = requests.get(shopify_url)
        
        if response.status_code == 200:
            collections = response.json().get('collections', [])
            target_collection = None
            
            for collection in collections:
                if collection.get('handle') == handle:
                    target_collection = collection
                    break
            
            if target_collection:
                return jsonify({
                    'success': True,
                    'title': target_collection.get('title', 'Unknown Collection'),
                    'description': target_collection.get('body_html', '').replace('<p>', '').replace('</p>', '').strip()[:200] + '...'
                })
            else:
                return jsonify({'success': False, 'error': 'Collection not found'})
        else:
            return jsonify({'success': False, 'error': 'Failed to fetch collection data'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/generate_ideas', methods=['POST'])
def generate_ideas():
    """Generate blog ideas based on uploaded CSV and settings"""
    try:
        # Get form data
        csv_file = request.files.get('csv_file')
        collection_url = request.form.get('collection_url')
        ai_model = request.form.get('ai_model')
        collection_type = request.form.get('collection_type')
        
        if not csv_file:
            return jsonify({'success': False, 'error': 'No CSV file uploaded'})
        
        # Create uploads directory if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save and process CSV using built-in csv module
        filename = secure_filename(csv_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        csv_file.save(filepath)
        
        # Read CSV and extract product data using built-in CSV reader
        unique_products = []
        seen_handles = set()
        
        with open(filepath, 'r', encoding='utf-8') as csvfile:
            # Detect delimiter and read CSV
            sample = csvfile.read(1024)
            csvfile.seek(0)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            
            reader = csv.DictReader(csvfile, delimiter=delimiter)
            
            for row in reader:
                handle = row.get('Handle', '')
                if handle and handle not in seen_handles:
                    seen_handles.add(handle)
                    unique_products.append({
                        'handle': handle,
                        'title': row.get('Title', ''),
                        'body': row.get('Body (HTML)', ''),
                        'tags': row.get('Tags', ''),
                        'image_url': row.get('Variant Image', ''),
                        'price': row.get('Variant Price', '')
                    })
        
        # Generate blog ideas based on collection type
        blog_ideas = generate_blog_ideas_for_type(collection_type, unique_products, collection_url)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'ideas': blog_ideas,
            'product_count': len(unique_products)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def generate_blog_ideas_for_type(collection_type, products, collection_url):
    """Generate blog ideas based on collection type"""
    
    # Extract collection name from URL
    collection_name = collection_url.split('/collections/')[-1].replace('-', ' ').title()
    
    if collection_type == 'business':
        return [
            {
                'title': f'Best {collection_name} for Small Businesses That Increase Foot Traffic',
                'description': 'Marketing-focused guide featuring top products for business growth',
                'category': 'Marketing',
                'wordCount': '800-1000',
                'type': 'marketing'
            },
            {
                'title': f'How to Choose {collection_name} for Your Business (Complete Guide)',
                'description': 'Step-by-step guide for business owners making purchasing decisions',
                'category': 'How-To',
                'wordCount': '1000-1200',
                'type': 'howto'
            },
            {
                'title': f'10 Creative {collection_name} Ideas That Actually Work',
                'description': 'Curated list of creative implementations with real examples',
                'category': 'Listicle',
                'wordCount': '800-1000',
                'type': 'listicle'
            },
            {
                'title': f'{collection_name} Psychology: How Customers Decide to Enter',
                'description': 'Deep dive into customer psychology and decision-making',
                'category': 'Psychology',
                'wordCount': '900-1100',
                'type': 'psychology'
            },
            {
                'title': f'Instagram-Worthy {collection_name} That Go Viral',
                'description': 'Social media focused content with shareable examples',
                'category': 'Social Media',
                'wordCount': '700-900',
                'type': 'social'
            },
            {
                'title': f'Why Generic {collection_name} Don\'t Work (And What Does)',
                'description': 'Problem-solving approach highlighting common mistakes',
                'category': 'Problem-Solving',
                'wordCount': '800-1000',
                'type': 'problem'
            },
            {
                'title': f'{collection_name} Trends Dominating 2025',
                'description': 'Current trends and future predictions in the industry',
                'category': 'Trends',
                'wordCount': '700-900',
                'type': 'trends'
            }
        ]
    
    elif collection_type == 'wedding':
        return [
            {
                'title': f'30 {collection_name} Ideas That Will Make Guests Swoon',
                'description': 'Romantic inspiration gallery with real wedding examples',
                'category': 'Inspiration',
                'wordCount': '1000-1200',
                'type': 'inspiration'
            },
            {
                'title': f'How to Choose {collection_name} That Match Your Wedding Theme',
                'description': 'Planning guide for coordinating with wedding aesthetics',
                'category': 'Planning',
                'wordCount': '900-1100',
                'type': 'planning'
            },
            {
                'title': f'DIY {collection_name}: Create Your Perfect Wedding Day',
                'description': 'Step-by-step DIY instructions and tips',
                'category': 'DIY',
                'wordCount': '1100-1300',
                'type': 'diy'
            },
            {
                'title': f'Affordable {collection_name} That Look Expensive',
                'description': 'Budget-friendly options that maintain premium appearance',
                'category': 'Budget',
                'wordCount': '800-1000',
                'type': 'budget'
            },
            {
                'title': f'Instagram-Perfect {collection_name} for Wedding Photos',
                'description': 'Photography-focused guide for social media worthy shots',
                'category': 'Photography',
                'wordCount': '700-900',
                'type': 'photography'
            }
        ]
    
    elif collection_type == 'home':
        return [
            {
                'title': f'{collection_name} That Transform Any Room',
                'description': 'Interior design focused guide with before/after examples',
                'category': 'Design',
                'wordCount': '900-1100',
                'type': 'design'
            },
            {
                'title': f'How to Style {collection_name} in Modern Homes',
                'description': 'Contemporary styling tips and placement strategies',
                'category': 'Styling',
                'wordCount': '800-1000',
                'type': 'styling'
            },
            {
                'title': f'Cozy {collection_name} Ideas for Every Season',
                'description': 'Seasonal decoration strategies and mood creation',
                'category': 'Seasonal',
                'wordCount': '800-1000',
                'type': 'seasonal'
            }
        ]
    
    elif collection_type == 'kids':
        return [
            {
                'title': f'{collection_name} That Spark Imagination',
                'description': 'Creative and educational focus for child development',
                'category': 'Creative',
                'wordCount': '800-1000',
                'type': 'creative'
            },
            {
                'title': f'Safe {collection_name} for Kids Bedrooms',
                'description': 'Safety-focused guide for parents with peace of mind',
                'category': 'Safety',
                'wordCount': '900-1100',
                'type': 'safety'
            }
        ]
    
    # Default fallback
    return [
        {
            'title': f'Best {collection_name} for Your Needs',
            'description': 'General guide covering top products and use cases',
            'category': 'Guide',
            'wordCount': '800-1000',
            'type': 'guide'
        }
    ]

@app.route('/publish_blog', methods=['POST'])
def publish_blog():
    """Generate and publish a single blog post"""
    try:
        data = request.json
        idea = data.get('idea')
        collection_url = data.get('collection_url')
        ai_model = data.get('ai_model', 'claude')
        
        # Generate the blog content
        blog_html = generate_blog_content(idea, collection_url, ai_model)
        
        # Create SEO-friendly slug
        slug = create_slug(idea['title'])
        
        # Prepare blog data for Shopify
        blog_data = {
            "article": {
                "title": idea['title'],
                "body_html": blog_html,
                "blog_id": BLOG_ID,
                "tags": get_smart_tags(idea['title'], idea['category']),
                "template_suffix": "ecom-neonxpert-blog-post",
                "published": True,
                "handle": slug,
                "metafields": [
                    {
                        "key": "title_tag",
                        "value": idea['title'][:70],
                        "value_type": "string",
                        "namespace": "global",
                        "type": "single_line_text_field"
                    },
                    {
                        "key": "description_tag",
                        "value": idea['description'],
                        "value_type": "string",
                        "namespace": "global",
                        "type": "single_line_text_field"
                    }
                ]
            }
        }
        
        # Upload to Shopify
        response = requests.post(
            f"{SHOP_URL}/blogs/{BLOG_ID}/articles.json",
            json=blog_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 201:
            article = response.json()['article']
            blog_url = f"https://{SHOP_NAME}.myshopify.com/blogs/neon-sign-ideas/{slug}"
            
            return jsonify({
                'success': True,
                'blog_id': article['id'],
                'blog_url': blog_url
            })
        else:
            return jsonify({
                'success': False,
                'error': f"Shopify API error: {response.status_code} - {response.text}"
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def generate_blog_content(idea, collection_url, ai_model):
    """Generate blog content using specified AI model with LATEST MODELS"""
    
    prompt = f"""
    Write a comprehensive, high-quality blog post with the title: "{idea['title']}"
    
    Requirements:
    - Category: {idea['category']}
    - Word count: {idea['wordCount']}
    - Target audience: Business owners and consumers
    - Include internal links to: {collection_url}
    - Include link to custom neon sign: https://neonxpert.com/products/custom-neon-sign
    - Write in professional, engaging tone that builds expertise and trust
    - Include practical advice and actionable tips
    - Use HTML formatting with proper headings (h2, h3)
    - Include bullet points and numbered lists where appropriate
    - Add a compelling conclusion with clear call-to-action
    
    CRITICAL: Follow E-E-A-T principles:
    - Experience: Include real insights and first-hand knowledge
    - Expertise: Demonstrate deep understanding of the topic
    - Authoritativeness: Reference industry standards and best practices
    - Trustworthiness: Provide accurate, helpful information
    
    Focus on providing genuine value and expertise while naturally incorporating product mentions.
    Write as if you're an expert in neon signage and business marketing.
    """
    
    if ai_model == 'claude':
        # Initialize Claude client only when needed - LATEST CLAUDE 3.5 SONNET
        global claude_client
        if claude_client is None:
            claude_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",  # LATEST AND BEST Claude model
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    else:  # ChatGPT - LATEST GPT-4o MODEL
        global openai_client
        if openai_client is None:
            openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
        response = openai_client.chat.completions.create(
            model="gpt-4o",  # LATEST AND BEST OpenAI model
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.7
        )
        return response.choices[0].message.content

def create_slug(title):
    """Create SEO-friendly slug from title"""
    import re
    slug = title.lower()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[-\s]+', '-', slug)
    return slug.strip('-')

def get_smart_tags(title, category):
    """Generate smart tags based on title and category"""
    tags = ["AutoBlog", category]
    
    title_lower = title.lower()
    if any(x in title_lower for x in ["business", "commercial"]): 
        tags.append("Business")
    if any(x in title_lower for x in ["wedding", "marriage"]):
        tags.append("Wedding")
    if any(x in title_lower for x in ["home", "decor", "room"]):
        tags.append("Home Decor")
    if any(x in title_lower for x in ["kids", "children", "family"]):
        tags.append("Kids")
    if any(x in title_lower for x in ["open", "sign"]):
        tags.append("Open Signs")
    
    return ", ".join(tags)

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
