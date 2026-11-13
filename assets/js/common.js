/* Restored interactions for the original Satellite Jekyll layout. */
(function(){
  'use strict';
  const $=(s,scope=document)=>scope.querySelector(s);
  const $$=(s,scope=document)=>[...scope.querySelectorAll(s)];
  document.addEventListener('DOMContentLoaded',function(){
    const loading=$('#loading'); if(loading){loading.classList.add('is-hidden');}
    const saved=localStorage.getItem('theme')==='dark';
    const setTheme=(dark)=>{
      document.body.classList.toggle('dark-theme',dark);
      $$('.ico-dark').forEach(x=>x.classList.toggle('active',dark));
      $$('.ico-light').forEach(x=>x.classList.toggle('active',dark));
      localStorage.setItem('theme',dark?'dark':'light');
    };
    setTheme(saved);
    $$('#btn-brightness').forEach(b=>b.addEventListener('click',()=>setTheme(!document.body.classList.contains('dark-theme'))));
    $('#btn-nav')?.addEventListener('click',function(){ $('#navigation')?.classList.toggle('nav-open'); this.setAttribute('aria-pressed',String($('#navigation')?.classList.contains('nav-open'))); });
    $$('.nav-list-expander').forEach(btn=>btn.addEventListener('click',function(){const li=this.closest('li');li?.classList.toggle('active');this.setAttribute('aria-pressed',String(li?.classList.contains('active'))); }));
    // Show nested navigation for the current article and keep original links usable.
    const active=$$('.nav-list-link').find(a=>a.pathname.replace(/\/$/,'')===location.pathname.replace(/\/$/,''));
    active?.closest('li')?.classList.add('current');
    let ancestor=active?.closest('li')?.parentElement?.closest('li');while(ancestor){ancestor.classList.add('active');ancestor=ancestor.parentElement?.closest('li');}
    const search=$('#search');
    $$('#btn-search').forEach(b=>b.addEventListener('click',()=>{search?.classList.add('active');$('#search-input')?.focus();}));
    search?.addEventListener('click',e=>{if(e.target===search || e.target.classList.contains('wave'))search.classList.remove('active');});
    document.addEventListener('keydown',e=>{if(e.key==='Escape')search?.classList.remove('active');});
    $('#btn-clear')?.addEventListener('click',()=>{const i=$('#search-input');if(i){i.value='';i.dispatchEvent(new Event('input'));i.focus();}});
    $('.top-arrow')?.addEventListener('click',()=>window.scrollTo({top:0,behavior:'smooth'}));
  });
  window.searchPost=function(posts){
    const input=document.getElementById('search-input'),results=document.getElementById('search-result');
    if(!input||!results)return;
    const esc=t=>String(t||'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    input.addEventListener('input',()=>{
      const q=input.value.toLowerCase().trim();results.replaceChildren();if(!q)return;
      posts.filter(p=>(p.type==='post')&&((p.title+' '+p.tags+' '+p.path).toLowerCase().includes(q))).slice(0,16).forEach(p=>{
        const li=document.createElement('li');const a=document.createElement('a');a.href=p.url;a.textContent=p.title;li.appendChild(a);results.appendChild(li);
      });
      if(!results.children.length)results.innerHTML='<li>No posts found</li>';
    });
  };
  window.searchRelated=function(posts){
    const ul=document.getElementById('related-posts');if(!ul)return;
    const others=posts.filter(p=>p.type==='post'&&p.url!==location.pathname).slice(0,3);
    others.forEach(p=>{const li=document.createElement('li');const a=document.createElement('a');a.href=p.url;a.className='title';a.textContent=p.title;li.appendChild(a);ul.appendChild(li);});
  };
})();
