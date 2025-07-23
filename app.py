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

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'nx-blog-generator-2025')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max

# API Clients
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
    try:
        data = request.json
        collection_url = data.get('url', '')
        
        if '/collections/' in collection_url:
            handle = collection_url.split('/collections/')[-1].strip('/')
        else:
            return jsonify({'success': False, 'error': 'Invalid collection URL'})
        
        custom_url = f"{SHOP_URL}/custom_collections.json"
        custom_response = requests.get(custom_url, headers=SHOPIFY_HEADERS)
        
        smart_url = f"{SHOP_URL}/smart_collections.json"
        smart_response = requests.get(smart_url, headers=SHOPIFY_HEADERS)
        
        target_collection = None
        
        if custom_response.status_code == 200:
            collections = custom_response.json().get('custom_collections', [])
            for collection in collections:
                if collection.get('handle') == handle:
                    target_collection = collection
                    break
        
        if not target_collection and smart_response.status_code == 200:
            collections = smart_response.json().get('smart_collections', [])
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
            error = 'Collection not found'
            if custom_response.status_code != 200:
                error += f' (Custom: {custom_response.status_code} - {custom_response.text})'
            if smart_response.status_code != 200:
                error += f' (Smart: {smart_response.status_code} - {smart_response.text})'
            return jsonify({'success': False, 'error': error})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/generate_topics', methods=['POST'])
def generate_topics():
    try:
        csv_file = request.files.get('csv_file')
        collection_url = request.form.get('collection_url')
        secondary_url = request.form.get('secondary_url', '')  # Optional
        ai_model = request.form.get('ai_model')
        
        if not csv_file:
            return jsonify({'success': False, 'error': 'No CSV file uploaded'})
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        filename = secure_filename(csv_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        csv_file.save(filepath)
        
        # Extract unique products (first row per handle)
        unique_products = {}
        
        with open(filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                handle = row.get('Handle', '')
                if handle and handle not in unique_products:
                    unique_products[handle] = {
                        'handle': handle,
                        'title': row.get('Title', ''),
                        'body': row.get('Body (HTML)', ''),
                        'tags': row.get('Tags', ''),
                        'image_url': row.get('Image Src', ''),
                        'price': row.get('Variant Price', '')
                    }
        
        product_data_json = json.dumps(list(unique_products.values()))
        
        # Generate topics
        blog_topics = generate_ai_topics(product_data_json, collection_url, secondary_url, ai_model)
        
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'topics': blog_topics,
            'product_count': len(unique_products)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def generate_ai_topics(product_data_json, collection_url, secondary_url, ai_model):
    collection_name = collection_url.split('/collections/')[-1].replace('-', ' ').title()
    
    secondary_prompt = f"Include 1-2 natural links to {secondary_url} if provided." if secondary_url else ""
    
    prompt = f"""
    Output ONLY a valid JSON array of topics, no additional text or explanations. Example: [{{"title": "Example", ...}}]
    
    Analyze this NeonXpert product data from the {collection_name} collection: {product_data_json}
    
    Requirements:
    - Identify subcategories/clusters from tags/titles (e.g., industries like coffee shops, dispensaries; intents like funny, budget).
    - Infer a keyword pool like Google autocomplete (e.g., 'neon open signs for cafes 2025', 'custom neon open signs weed').
    - Generate a dynamic number of unique blog topics/titles based on scope (e.g., 2-5 for small, 20+ for large with 100+ products; 1-2 pillars + cluster-specific).
    - Each topic: {{"title": "SEO-optimized title with NeonXpert mention", "description": "Brief summary", "category": "e.g., How-To", "wordCount": "Dynamic: 800-3000 based on scope", "type": "e.g., listicle", "relevant_products": [list of handles for links/images]}}
    - Variety: Mix listicles, guides, trends; avoid repetition.
    - Value: Practical, data-backed (use real stats like 'signs boost traffic 15-30% per studies').
    """
    
    if ai_model == 'claude':
        global claude_client
        if claude_client is None:
            claude_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=8000,
            messages=[{"role": "user", "content": prompt}]
        )
        # Clean and parse
        response_text = response.content[0].text.strip()
        if response_text.startswith('[') and response_text.endswith(']'):
            return json.loads(response_text)
        else:
            return json.loads('[' + response_text + ']')  # Fallback if not array
    else:
        global openai_client
        if openai_client is None:
            openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8000,
            temperature=0.7
        )
        response_text = response.choices[0].message.content.strip()
        if response_text.startswith('[') and response_text.endswith(']'):
            return json.loads(response_text)
        else:
            return json.loads('[' + response_text + ']')  # Fallback

@app.route('/publish_blog', methods=['POST'])
def publish_blog():
    try:
        data = request.json
        topic = data.get('topic')
        collection_url = data.get('collection_url')
        secondary_url = data.get('secondary_url', '')
        ai_model = data.get('ai_model')
        product_data = data.get('product_data')  # Full unique products from frontend
        
        blog_html = generate_blog_content(topic, collection_url, secondary_url, product_data, ai_model)
        
        featured_image_url = topic['relevant_products'][0]['image_url'] if topic['relevant_products'] else ''
        featured_image_id = upload_image_to_shopify(featured_image_url)
        
        slug = create_slug(topic['title'])
        
        blog_data = {
            "article": {
                "title": topic['title'],
                "body_html": blog_html,
                "blog_id": BLOG_ID,
                "tags": get_smart_tags(topic['title'], topic['category']),
                "template_suffix": "ecom-neonxpert-blog-post",
                "published": True,
                "handle": slug,
                "image": {"src": featured_image_id} if featured_image_id else None,
                "metafields": [
                    {"key": "title_tag", "value": topic['title'][:70], "type": "single_line_text_field", "namespace": "global"},
                    {"key": "description_tag", "value": topic['description'], "type": "single_line_text_field", "namespace": "global"}
                ]
            }
        }
        
        response = requests.post(
            f"{SHOP_URL}/blogs/{BLOG_ID}/articles.json",
            json=blog_data,
            headers=SHOPIFY_HEADERS
        )
        
        if response.status_code == 201:
            article = response.json()['article']
            blog_url = f"https://{SHOP_NAME}.myshopify.com/blogs/neon-sign-ideas/{slug}"
            return jsonify({'success': True, 'blog_id': article['id'], 'blog_url': blog_url})
        else:
            return jsonify({'success': False, 'error': f"Shopify API error: {response.status_code} - {response.text}"})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def generate_blog_content(topic, collection_url, secondary_url, product_data, ai_model):
    product_json = json.dumps(product_data)
    
    secondary_prompt = f"Include 1-2 natural links to {secondary_url} (e.g., 'Customize at NeonXpert's custom neon sign page') if provided." if secondary_url else ""
    
    prompt = f"""
    Write a high-quality blog post titled: "{topic['title']}"
    
    Requirements:
    - Word count: Dynamic (800-3000 based on scope; e.g., shorter for niche, longer for guides).
    - Use product data: {product_json} – Mention relevant NeonXpert products naturally with links (e.g., <a href="https://neonxpert.com/products/{handle}">{title}</a>); no prices unless topic is price-focused.
    - Embed images: For listicles, add <img src="{image_url}" alt="NeonXpert {title} - optimized keyword" style="max-width: 100%; height: auto; display: block; margin: 0 auto;" /> after items. Featured: First product.
    - Structure: HTML with h2/h3, short paragraphs, bullets/lists for tips; value bombs (e.g., pro tips, stats like 'signs boost traffic 15-30% per studies').
    - Tone: Engaging NeonXpert expert; 5-7 mentions.
    - Interlinks: After writing, embed naturally: 2-3 to {collection_url} (e.g., 'Explore NeonXpert's open neon sign collection'); {secondary_prompt}.
    - Factual: Use real stats (e.g., 'Illuminated signs increase walk-ins by 15% – Sign Research 2025'); optional FAQ if natural.
    - Optimize: Keywords in headings; modular for LLMs.
    """
    
    if ai_model == 'claude':
        global claude_client
        if claude_client is None:
            claude_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=8000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    else:
        global openai_client
        if openai_client is None:
            openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8000,
            temperature=0.7
        )
        return response.choices[0].message.content

def upload_image_to_shopify(image_url):
    if not image_url:
        return None
    
    image_response = requests.get(image_url)
    if image_response.status_code != 200:
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
        }
    }
    """
    variables = {
        "input": [{
            "filename": os.path.basename(image_url),
            "mimeType": "image/jpeg",
            "httpMethod": "POST",
            "resource": "IMAGE"
        }]
    }
    
    response = requests.post(graphql_url, json={"query": query, "variables": variables}, headers=SHOPIFY_HEADERS)
    if response.status_code == 200:
        staged_data = response.json()['data']['stagedUploadsCreate']['stagedTargets'][0]
        upload_url = staged_data['url']
        params = {p['name']: p['value'] for p in staged_data['parameters']}
        
        files = {'file': (os.path.basename(image_url), image_response.content)}
        upload_response = requests.post(upload_url, data=params, files=files)
        
        if upload_response.status_code == 200 or upload_response.status_code == 201:
            return staged_data['resourceUrl']
    return None

def create_slug(title):
    import re
    slug = title.lower()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[-\s]+', '-', slug)
    return slug.strip('-')

def get_smart_tags(title, category):
    tags = ["AutoBlog", category]
    
    title_lower = title.lower()
    if "business" in title_lower or "commercial" in title_lower: 
        tags.append("Business")
    if "wedding" in title_lower or "marriage" in title_lower:
        tags.append("Wedding")
    if "home" in title_lower or "decor" in title_lower or "room" in title_lower:
        tags.append("Home Decor")
    if "kids" in title_lower or "children" in title_lower or "family" in title_lower:
        tags.append("Kids")
    if "open" in title_lower and "sign" in title_lower:
        tags.append("Open Signs")
    
    return ", ".join(tags)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
