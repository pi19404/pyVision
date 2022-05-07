module Jekyll

    
    # class Lowercase < BibTeX::Filter
    #       def apply(value)
    #         # Use of \g<1> pattern back-reference to allow for capturing nested {} groups.
    #         # The first (outermost) capture of $1 is used.
    #         value.to_s.gsub(/\\lowercase(\{(?:[^{}]|\g<1>)*\})/) {
    #           "#{$1[1..-2].downcase | remove:' '}"
    #         }
    #       end
    # end
    

    class CategoryPage < Page
      def initialize(site, base, dir, category)
        @site = site
        @base = base
        @dir = dir
        @name = 'index.html'
  
        self.process(@name)
        self.read_yaml(File.join(base, '_layouts'), 'category.html')
        #self.data['category'] = category
        cat4url =category.downcase
        cat4url = Utils.slugify(cat4url)
        #cat4url =  #{ $cat4url | remove:' ' }
        category_title_prefix = site.config['category_title_prefix'] || 'Category: '
        self.data['title'] = "#{category_title_prefix}#{category}"
        self.data['category-name']=category
        self.data['permalink']="/category1/"+cat4url
      end
    end
  
    class CategoryPageGenerator < Generator
      safe true
  
      def generate(site)
        if site.layouts.key? 'category'
          dir = site.config['category_dir'] || 'category1'
          
          for category in site.categories
          #site.categories.each_key do |category|
            cat4url =  category[0]  
            site.pages << CategoryPage.new(site, site.source, File.join(dir, cat4url ), category[0])
          end
        end
      end
    end
  end