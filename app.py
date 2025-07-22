import os
import csv
import json
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import anthropic
import openai
from werkzeug.utils import secure_filename

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'nx-blog-generator-2025')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# API Clients
claude_client = None
openai_client = None

# MODERN SHOPIFY CONFIG - 2025 APPROACH
SHOP_NAME = os.getenv("SHOP_NAME")
ACCESS_TOKEN = os.getenv("SHOPIFY_ACCESS_TOKEN")
BLOG_ID = os.getenv("SHOPIFY_BLOG_ID")
SHOP_URL = f"https://{SHOP_NAME}.myshopify.com/admin/api/2025-01"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/validate_collection', methods=['POST'])
def validate_collection():
    """Validate collection URL using MODERN Shopify API"""
    try:
        data = request.json
        collection_url = data.get('url', '')
        
        if '/collections/' in collection_url:
            handle = collection_url.split('/collections/')[-1].strip('/')
        else:
            return jsonify({'success': False, 'error': 'Invalid collection URL format'})
        
        # MODERN SHOPIFY API HEADERS
        headers = {
            "X-Shopify-Access-Token": ACCESS_TOKEN,
            "Content-Type": "application/json"
        }
        
        target_collection = None
        
        # Check custom collections first
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
            print(f"Custom collections error: {e}")
        
        # Check smart collections if not found
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
                print(f"Smart collections error: {e}")
        
        if target_collection:
            description = target_collection.get('body_html', '')
            if description:
                # Clean HTML tags
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
                'error': f'Collection "{handle}" not found. Please check the URL and try again.'
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': f'Validation error: {str(e)}'})

@app.route('/generate_ideas', methods=['POST'])
def generate_ideas():
    """Generate blog ideas based on uploaded CSV and settings"""
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
            # Auto-detect delimiter
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
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'ideas': blog_ideas,
            'product_count': len(unique_products)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Generation error: {str(e)}'})

def generate_blog_ideas_for_type(collection_type, products, collection_url):
    """Generate SEO-optimized blog ideas based on collection type"""
    
    collection_name = collection_url.split('/collections/')[-1].replace('-', ' ').title()
    
    if collection_type == 'business' or 'open' in collection_name.lower():
        return [
            {
                'title': f'Best {collection_name} for Small Businesses That Increase Foot Traffic (2025)',
                'description': 'Comprehensive guide to choosing business signage that drives customer engagement and sales',
                'category': 'Business Marketing',
                'wordCount': '1200-1500',
                'type': 'marketing'
            },
            {
                'title': f'How to Choose {collection_name} for Your Business (Complete Guide)',
                'description': 'Expert buyer guide with practical tips for selecting the perfect business signage',
                'category': 'How-To Guide',
                'wordCount': '1400-1700',
                'type': 'howto'
            },
            {
                'title': f'10 {collection_name} Ideas That Actually Boost Sales',
                'description': 'Data-driven examples of effective business signage with real conversion results',
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
                'description': 'Social media optimization guide for business signage that generates organic shares',
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
                'description': 'Romantic wedding signage inspiration with real examples from recent weddings',
                'category': 'Wedding Inspiration',
                'wordCount': '1500-1800',
                'type': 'inspiration'
            },
            {
                'title': f'How to Choose {collection_name} That Match Your Wedding Theme Perfectly',
                'description': 'Wedding planning guide for selecting signage that complements your special day',
                'category': 'Wedding Planning',
                'wordCount': '1200-1500',
                'type': 'planning'
            },
            {
                'title': f'DIY {collection_name} vs Professional: What Every Couple Should Know',
                'description': 'Cost comparison and quality analysis for wedding signage options',
                'category': 'Wedding DIY',
                'wordCount': '1100-1400',
                'type': 'comparison'
            },
            {
                'title': f'Budget-Friendly {collection_name} That Look Expensive',
                'description': 'Affordable wedding signage options that maintain premium appearance',
                'category': 'Wedding Budget',
                'wordCount': '1000-1300',
                'type': 'budget'
            },
            {
                'title': f'Instagram-Perfect {collection_name} for Unforgettable Wedding Photos',
                'description': 'Photography-focused guide for creating social media worthy wedding moments',
                'category': 'Wedding Photography',
                'wordCount': '900-1200',
                'type': 'photography'
            }
        ]
    
    elif collection_type == 'home':
        return [
            {
                'title': f'{collection_name} That Transform Any Room: Interior Design Guide',
                'description': 'Complete home decoration guide with before/after examples and styling tips',
                'category': 'Home Decor',
                'wordCount': '1200-1500',
                'type': 'design'
            },
            {
                'title': f'How to Style {collection_name} in Modern Homes (2025 Trends)',
                'description': 'Contemporary home styling guide with placement strategies and color coordination',
                'category': 'Interior Styling',
                'wordCount': '1000-1300',
                'type': 'styling'
            },
            {
                'title': f'Seasonal {collection_name} Ideas for Year-Round Home Decor',
                'description': 'Creative seasonal decoration strategies and mood-setting techniques',
                'category': 'Seasonal Decor',
                'wordCount': '1100-1400',
                'type': 'seasonal'
            }
        ]
    
    elif collection_type == 'kids':
        return [
            {
                'title': f'{collection_name} That Spark Imagination: Creative Kids Room Ideas',
                'description': 'Child development focused decor guide with educational and fun elements',
                'category': 'Kids Room Decor',
                'wordCount': '1000-1300',
                'type': 'creative'
            },
            {
                'title': f'Safe {collection_name} for Kids Bedrooms: Parent\'s Complete Guide',
                'description': 'Safety-first approach to kids room decoration with peace of mind tips',
                'category': 'Child Safety',
                'wordCount': '1100-1400',
                'type': 'safety'
            }
        ]
    
    else:
        # Custom collection type
        return [
            {
                'title': f'Ultimate {collection_name} Guide: Everything You Need to Know (2025)',
                'description': f'Comprehensive guide covering everything about {collection_name} with expert insights',
                'category': 'Complete Guide',
                'wordCount': '1400-1700',
                'type': 'guide'
            },
            {
                'title': f'How to Choose the Perfect {collection_name} for Your Needs',
                'description': 'Expert selection guide with practical tips and decision framework',
                'category': 'Buying Guide',
                'wordCount': '1200-1500',
                'type': 'howto'
            },
            {
                'title': f'10 Creative {collection_name} Ideas That Will Inspire You',
                'description': 'Innovative ideas and creative implementations with visual examples',
                'category': 'Creative Ideas',
                'wordCount': '1000-1300',
                'type': 'inspiration'
            },
            {
                'title': f'{collection_name} Trends You Need to Know in 2025',
                'description': 'Latest trends, innovations, and what to expect this year',
                'category': 'Trends & Innovation',
                'wordCount': '900-1200',
                'type': 'trends'
            },
            {
                'title': f'Instagram-Worthy {collection_name} That Get Thousands of Likes',
                'description': 'Social media optimization guide for maximum engagement and shares',
                'category': 'Social Media',
                'wordCount': '800-1100',
                'type': 'social'
            }
        ]

@app.route('/publish_blog', methods=['POST'])
def publish_blog():
    """Publish blog using MODERN Shopify API"""
    try:
        data = request.json
        idea = data.get('idea')
        collection_url = data.get('collection_url')
        ai_model = data.get('ai_model', 'claude')
        
        # Generate blog content
        blog_html = generate_blog_content(idea, collection_url, ai_model)
        
        if not blog_html or 'generation failed' in blog_html.lower():
            return jsonify({'success': False, 'error': 'Failed to generate blog content'})
        
        # Create slug
        slug = create_slug(idea['title'])
        
        # MODERN SHOPIFY API HEADERS
        headers = {
            "X-Shopify-Access-Token": ACCESS_TOKEN,
            "Content-Type": "application/json"
        }
        
        # Prepare blog data
        blog_data = {
            "article": {
                "title": idea['title'],
                "body_html": blog_html,
                "blog_id": int(BLOG_ID),
                "tags": get_smart_tags(idea['title'], idea['category']),
                "published": True,
                "handle": slug,
                "summary": idea['description'][:150] + "...",
                "metafields": [
                    {
                        "key": "title_tag",
                        "value": f"{idea['title']} | NeonXpert - Premium Neon Signs",
                        "value_type": "string",
                        "namespace": "global"
                    },
                    {
                        "key": "description_tag", 
                        "value": f"{idea['description']} Shop NeonXpert's premium collection today.",
                        "value_type": "string",
                        "namespace": "global"
                    }
                ]
            }
        }
        
        # Publish to Shopify
        publish_url = f"{SHOP_URL}/blogs/{BLOG_ID}/articles.json"
        response = requests.post(publish_url, json=blog_data, headers=headers, timeout=30)
        
        if response.status_code == 201:
            article = response.json()['article']
            blog_url = f"https://{SHOP_NAME}.myshopify.com/blogs/neon-sign-ideas/{slug}"
            
            return jsonify({
                'success': True,
                'blog_id': article['id'],
                'blog_url': blog_url,
                'title': article['title']
            })
        else:
            error_msg = f"Shopify API Error {response.status_code}: {response.text[:200]}"
            return jsonify({'success': False, 'error': error_msg})
            
    except Exception as e:
        return jsonify({'success': False, 'error': f'Publishing error: {str(e)}'})

def generate_blog_content(idea, collection_url, ai_model):
    """Generate high-quality blog content using AI"""
    
    prompt = f"""
    Write a comprehensive, SEO-optimized blog post with the title: "{idea['title']}"
    
    REQUIREMENTS:
    - Category: {idea['category']}
    - Target word count: {idea['wordCount']}
    - Write for NeonXpert customers and prospects
    - Include internal link to: {collection_url}
    - Include custom neon link: https://neonxpert.com/products/custom-neon-sign
    - Use proper HTML structure with H2, H3, lists, and paragraphs
    - Include compelling introduction and strong conclusion with CTA
    - Write in professional, engaging tone that demonstrates expertise
    - Include practical tips and actionable advice
    - Add FAQ section for better SEO
    - Mention "NeonXpert" 3-5 times naturally
    
    CONTENT GUIDELINES:
    - Follow E-E-A-T principles (Experience, Expertise, Authoritativeness, Trustworthiness)
    - Include industry insights and expert knowledge
    - Use data and statistics where relevant
    - Write for both humans and search engines
    - Create scannable content with clear headings
    - End with strong call-to-action to visit the collection
    
    Write as a neon signage expert with deep industry knowledge and customer understanding.
    """
    
    try:
        if ai_model == 'claude':
            global claude_client
            if claude_client is None:
                claude_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            
            # Try latest Claude models
            models = ["claude-3-5-sonnet-20241022", "claude-3-sonnet-20240229"]
            
            for model in models:
                try:
                    response = claude_client.messages.create(
                        model=model,
                        max_tokens=4000,
                        temperature=0.7,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return response.content[0].text
                except Exception as e:
                    print(f"Claude model {model} failed: {e}")
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
            return response.choices[0].message.content
            
    except Exception as e:
        print(f"AI generation error: {e}")
        
    return f"<h2>Error generating content</h2><p>Please check your API configuration and try again.</p>"

def create_slug(title):
    """Create SEO-friendly URL slug"""
    import re
    slug = title.lower()
    slug = re.sub(r'[^\w\s-]', '', slug)  # Remove special chars
    slug = re.sub(r'[-\s]+', '-', slug)   # Replace spaces/hyphens
    return slug.strip('-')[:50]           # Limit length

def get_smart_tags(title, category):
    """Generate relevant tags for SEO and organization"""
    tags = ["AutoBlog", category, "NeonXpert", "2025"]
    
    title_lower = title.lower()
    
    # Add contextual tags
    if any(word in title_lower for word in ["business", "commercial", "store"]):
        tags.append("Business Signs")
    if any(word in title_lower for word in ["wedding", "marriage", "bride"]):
        tags.append("Wedding Decor")
    if any(word in title_lower for word in ["home", "room", "decor"]):
        tags.append("Home Decor")
    if any(word in title_lower for word in ["open", "entrance", "welcome"]):
        tags.append("Open Signs")
    if any(word in title_lower for word in ["instagram", "social", "viral"]):
        tags.append("Social Media")
    if any(word in title_lower for word in ["diy", "custom", "personalized"]):
        tags.append("Custom Signs")
    
    return ", ".join(tags)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
